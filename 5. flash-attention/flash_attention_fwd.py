import torch
import triton
import triton.language as tl
import math

# device 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=8, num_stages=2),
    ],
    key=['seq_len', 'head_dim'],
)
@triton.jit
def flash_attention_kernel(
    Q, K, V, O, L,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    seq_len,
    head_dim : tl.constexpr, 
    BLOCK_M : tl.constexpr,
    BLOCK_N : tl.constexpr
):
    pid_batch = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)
    pid_m = tl.program_id(axis=2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M) 
    offs_d = tl.arange(0, head_dim)
    q_ptrs = Q + pid_batch * stride_qb + pid_head * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd 
    q_mask = offs_m[:, None] < seq_len
    # finally loading q into mem
    q_tile = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # now initializing the running states
    m = tl.full((BLOCK_M,), value=-float('inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    o = tl.zeros((BLOCK_M, head_dim), dtype=tl.float32)

    for j in range(tl.cdiv(seq_len, BLOCK_N)):
        offs_n = j * BLOCK_N + tl.arange(0, BLOCK_N)
        k_ptrs = K + pid_batch * stride_kb + pid_head * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd 
        v_ptrs = V + pid_batch * stride_vb + pid_head * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd 

        k_mask = offs_n[:, None] < seq_len
        v_mask = offs_n[:, None] < seq_len

        k_tile = tl.load(k_ptrs, mask=k_mask, other=0.0)
        v_tile = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # compute S with the 1/root(d) scailing factor 
        num = tl.dot(q_tile, tl.trans(k_tile))
        # S = num / tl.sqrt(head_dim.to(tl.float32))
        S = num / (head_dim ** 0.5) # replaced the above one with this new one for precision issues
        S = S.to(tl.float32) # trying to force it to produce fp32 

        # S = tl.where(offs_n[None, :] < seq_len, S, float('-inf')) # by this we are replacing the mask values from 0 to -inf cuz there shoujdl be no involvement of 0
        # now we replace the above with padding + causal mask together 

        padding_ok = offs_n[None, :] < seq_len
        causal_ok = offs_m[:, None] >= offs_n[None, :]
        mask = padding_ok & causal_ok
        S = tl.where(mask, S, float('-inf'))
        # compute the running vars 
        # 1. rowmax 
        rowmax = tl.max(S, axis=1)
        # 2. max
        m_new = tl.maximum(m, rowmax)
        # 3. rowsum
        tilde_p = tl.exp(S - m_new[:, None])
        rowsum = tl.sum(tilde_p, axis=1) 
        l_new = tl.exp(m - m_new) * l + rowsum

        # now o
        rescale_factor = tl.exp(m - m_new)
        old_c = rescale_factor[:, None] * o # this broadcasts the per row scalar across head_dim
        new_c = tl.dot(tilde_p, v_tile.to(tl.float32)) # casting to fp32
        o_new = old_c + new_c

        # update the new values to the running states (vars)
        m = m_new
        l = l_new
        o = o_new

    # now out of the loop
    final_o = o / l[:, None]
    final_l = m + tl.log(l)

    # finally write the computed values back to hbm
    o = final_o
    l = final_l

    # finally write o back to output 
    o_ptrs = O + pid_batch * stride_ob + pid_head * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od 
    o_mask = offs_m[:, None] < seq_len
    tl.store(o_ptrs, final_o, mask=o_mask)

    l_ptrs = L + pid_batch * stride_lb + pid_head * stride_lh + offs_m * stride_ls
    l_mask = offs_m < seq_len
    tl.store(l_ptrs, final_l, mask=l_mask)

#This kernel beolw precomputes D_i = rowsum(dO_i . O_i) for every row, once, up front. 
# D_i is the per-row scalar the softmax-Jacobian shortcut needs (dS = P  . (dP − D)) 
# to avoid ever materializing the full Jacobian. Computing it here — from the fully-materialized O and dO 
# which means the block loop later can just load D_i per row instead of recomputing it inside every block iteration.

@triton.jit
def preprocess_kernel(
    O, dO, D,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_Db, stride_Dh, stride_Ds,
    seq_len,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    
    # 1. get pid_batch, pid_head, pid_m from program_id
    pid_batch = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)
    pid_m = tl.program_id(axis=2)
    # 2. build offs_m, offs_d (same as forward's setup)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, head_dim)
    # 3. build o_ptrs and do_ptrs (same pattern as q_ptrs in forward)
    o_ptrs = O + pid_batch * stride_ob + pid_head * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    o_mask = offs_m[:, None] < seq_len
    do_ptrs = dO + pid_batch * stride_dob + pid_head * stride_doh + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    do_mask = offs_m[:, None] < seq_len
    # 4. load o_tile and do_tile (masked, other=0.0)
    o_tile = tl.load(o_ptrs, mask=o_mask, other=0.0)
    do_tile = tl.load(do_ptrs, mask=do_mask, other=0.0)
    # 5. elementwise multiply, then tl.sum along axis=1 to get D_row (shape BLOCK_M,)
    prod = o_tile * do_tile
    D_row = tl.sum(prod, axis=1)    
    # 6. build D_ptrs (1D, like your fixed l_ptrs — no offs_d needed)
    D_ptrs = D + pid_batch * stride_Db + pid_head * stride_Dh + offs_m * stride_Ds
    D_mask = offs_m < seq_len
    # 7. store D_row into D_ptrs, masked
    tl.store(D_ptrs, D_row, mask=D_mask)


@triton.jit
def backward_dkdv_kernel(
    Q, K, V, O, dO, L, D, dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_Lb, stride_Lh, stride_Ls,
    stride_Db, stride_Dh, stride_Ds,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    seq_len,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,   # Q block size (inner loop)
    BLOCK_N: tl.constexpr,   # K/V block size (outer/fixed)
):
    pid_batch = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)   # <-- note: pid over K/V blocks now, not Q blocks

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # K/V rows (fixed for this whole kernel instance)
    offs_d = tl.arange(0, head_dim)

    # 1. Load K_block, V_block ONCE (outside the loop)
    k_ptrs = K + pid_batch * stride_kb + pid_head * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd 
    v_ptrs = V + pid_batch * stride_vb + pid_head * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd 

    k_mask = offs_n[:, None] < seq_len
    v_mask = offs_n[:, None] < seq_len

    k_tile = tl.load(k_ptrs, mask=k_mask, other=0.0)
    v_tile = tl.load(v_ptrs, mask=v_mask, other=0.0)
    
    # 2. Initialize dK_acc, dV_acc as zeros, shape (BLOCK_N, head_dim) — these accumulate across the Q loop.
    dK_acc = tl.zeros((BLOCK_N, head_dim), dtype=tl.float32)
    dV_acc = tl.zeros((BLOCK_N, head_dim), dtype=tl.float32)

    # 3. Loop over Q blocks
    for i in range(tl.cdiv(seq_len, BLOCK_M)):

        offs_m = i * BLOCK_M + tl.arange(0, BLOCK_M)
        # a. load Q_block, dO_block, L_block, D_block for this Q chunk 
        q_ptrs = Q + pid_batch * stride_qb + pid_head * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd 
        q_mask = offs_m[:, None] < seq_len
        q_tile = tl.load(q_ptrs, mask=q_mask, other=0.0)

        dO_ptrs = dO + pid_batch * stride_dob + pid_head * stride_doh + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
        dO_mask = offs_m[:, None] < seq_len
        dO_tile = tl.load(dO_ptrs, mask=dO_mask, other=0.0)

        l_ptrs = L + pid_batch * stride_Lb + pid_head * stride_Lh + offs_m * stride_Ls
        l_mask = offs_m < seq_len
        l_tile = tl.load(l_ptrs, mask=l_mask, other=0.0)

        D_ptrs = D + pid_batch * stride_Db + pid_head * stride_Dh + offs_m * stride_Ds
        D_mask = offs_m < seq_len
        D_tile = tl.load(D_ptrs, mask=D_mask, other=0.0)

        # b. S = Q_block @ K_block.T / sqrt(head_dim)
        S = tl.dot(q_tile, tl.trans(k_tile)) / (head_dim ** 0.5)
        # c. apply causal + padding mask (same logic as forward — careful: roles of offs_m/offs_n in the mask condition are the same comparison, just Q block vs K/V block)
        padding_ok = offs_n[None, :] < seq_len
        causal_ok = offs_m[:, None] >= offs_n[None, :]
        mask = padding_ok & causal_ok
        S = tl.where(mask, S, float('-inf'))
        # d. P = exp(S - L_block[:, None])
        P = tl.exp(S - l_tile[:, None])
        # e. dV_acc += P.T @ dO_block
        dV_acc += tl.dot(tl.trans(P), dO_tile.to(tl.float32))
        # f. dP = dO_block @ V_block.T
        dP = tl.dot(dO_tile, tl.trans(v_tile))
        # g. dS = P * (dP - D_block[:, None])
        dS = P * (dP - D_tile[:, None])
        # h. dK_acc += dS.T @ Q_block
        dK_acc += tl.dot(tl.trans(dS), q_tile.to(tl.float32))
    # 4. After the loop: store dK_acc into dK, dV_acc into dV (masked by offs_n < seq_len — this is a K/V-row mask now, not a Q-row mask)
    dK_ptrs = dK + pid_batch * stride_dkb + pid_head * stride_dkh + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dK_mask = offs_n[:, None] < seq_len
    tl.store(dK_ptrs, dK_acc, mask=dK_mask)

    dV_ptrs = dV + pid_batch * stride_dvb + pid_head * stride_dvh + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
    dV_mask = offs_n[:, None] < seq_len
    tl.store(dV_ptrs, dV_acc, mask=dV_mask)

def flash_attention_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):

    # extracting the size for q k v tensors
    batch, num_heads, seq_len, head_dim = Q.shape
    assert K.shape == Q.shape and V.shape == Q.shape, "Q, K, V shape mismatch"

    assert Q.is_cuda and K.is_cuda and V.is_cuda , "Not CUDA Tensors -_-"

    # output tensor
    O = torch.empty(batch, num_heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)
    L = torch.empty(batch, num_heads, seq_len, device=DEVICE, dtype=torch.float16)

    # define the launchpad grid
    grid = lambda META: (batch, num_heads, triton.cdiv(seq_len, META['BLOCK_M']))

    flash_attention_kernel[grid](
        Q, K, V, O, L,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        seq_len, head_dim,
    )

    return O, L

def preprocess_kernel_forward(O: torch.Tensor, dO: torch.Tensor):

    batch, num_heads, seq_len, head_dim = O.shape
    assert O.shape == dO.shape, "Q, K, V shape mismatch"
    assert O.is_cuda and dO.is_cuda, "Not CUDA Tensors -_-"

    D = torch.empty(batch, num_heads, seq_len, device=DEVICE, dtype=torch.float16)

    BLOCK_M = 64
    grid = (batch, num_heads, triton.cdiv(seq_len, BLOCK_M))

    preprocess_kernel[grid](
    O, dO, D,
    O.stride(0), O.stride(1), O.stride(2), O.stride(3),
    dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
    D.stride(0), D.stride(1), D.stride(2),
    seq_len, head_dim, BLOCK_M,
    )

    return D

def backward_dkdv_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                          O: torch.Tensor, dO: torch.Tensor, L: torch.Tensor,
                          D: torch.Tensor, dK: torch.Tensor, dV: torch.Tensor):

    batch, num_heads, seq_len, head_dim = K.shape

    BLOCK_M = 64
    BLOCK_N = 64

    grid = (batch, num_heads, triton.cdiv(seq_len, BLOCK_N))

    backward_dkdv_kernel[grid](
        Q, K, V, O, dO, L, D, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        D.stride(0), D.stride(1), D.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        seq_len, head_dim, BLOCK_M, BLOCK_N,
    )

    return dK, dV


def run_fa_fwd(batch, heads, seq_len, head_dim):

    torch.manual_seed(42)

    Q = torch.rand(batch, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)
    K = torch.rand(batch, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)
    V = torch.rand(batch, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)

    output_flash_attention_o, _ = flash_attention_forward(Q, K, V)

    # causal reference — PyTorch's built-in, is_causal=True applies the
    # same "key position <= query position" rule you just added
    output_torch = torch.nn.functional.scaled_dot_product_attention(
        Q, K, V, is_causal=True
    )

    diff = (output_flash_attention_o - output_torch).abs()
    print("max abs diff:", diff.max().item())
    print("mean abs diff:", diff.mean().item())

    result = torch.allclose(output_flash_attention_o, output_torch, atol=1e-2, rtol=1e-3)
    return result

def test_preprocess_kernel(batch, heads, seq_len, head_dim):
    torch.manual_seed(0)
    
    O = torch.rand(batch, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)
    dO = torch.rand(batch, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)

    D = preprocess_kernel_forward(O, dO).to(torch.float32)

    D_ref = (O.float() * dO.float()).sum(dim=-1)

    print("test preprocess kernel results  : \n ")
    diff = (D - D_ref).abs()
    print("max abs diff:", diff.max().item())
    print("mean abs diff:", diff.mean().item())
    result = torch.allclose(D, D_ref, atol=1e-2, rtol=1e-3)

    return result


def test_backward_dkdv_kernel(batch, heads, seq_len, head_dim):
    torch.manual_seed(42)

    Q = torch.rand(batch, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)
    K = torch.rand(batch, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)
    V = torch.rand(batch, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)
    O = torch.rand(batch, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)
    dO = torch.rand(batch, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)
    L = torch.rand(batch, heads, seq_len, device=DEVICE, dtype=torch.float16)
    D = torch.rand(batch, heads, seq_len, device=DEVICE, dtype=torch.float16)
    dK = torch.rand(batch, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)
    dV = torch.rand(batch, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)

    dK, dV = backward_dkdv_forward(
        Q, K, V,
        O, dO, L,
        D, dK, dV
    )

    dK = dK.to(torch.float32)
    dV = dV.to(torch.float32)

    Q_ref = Q.clone().requires_grad_(True)
    K_ref = K.clone().requires_grad_(True)
    V_ref = V.clone().requires_grad_(True)

    S = torch.matmul(Q_ref, torch.transpose(K_ref, 2, 3)) / (head_dim ** 0.5)

    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=DEVICE, dtype=torch.bool))
    S = S.masked_fill(~causal_mask, float('-inf'))
    P = torch.softmax(S, dim=-1)
    O_ref = torch.matmul(P, V_ref)
    O_ref.backward(dO)

# Peak FP32 for the MOBILE RTX 5080 (~32 TFLOP/s reference).
PEAK_TFLOPS = 49.15
def benchmark(batch, heads, head_dim):

    sizes = [512, 1024, 2048, 4096, 8192]
    print(f"{'Size':>6} | {'Triton':>10} | {'cuBLAS':>10} | {'Tri %pk':>8} | {'cuB %pk':>8} | {'Tri MB':>8} | {'cuB MB':>8}")
    print("-" * 90)

    for n in sizes:
        Q = torch.rand(batch, heads, n, head_dim, device=DEVICE, dtype=torch.float16)
        K = torch.rand(batch, heads, n, head_dim, device=DEVICE, dtype=torch.float16)
        V = torch.rand(batch, heads, n, head_dim, device=DEVICE, dtype=torch.float16)

        # warmup (also lets autotune settle on a config for this shape)
        flash_attention_forward(Q, K, V)
        torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)
        torch.cuda.synchronize()

        # --- Triton: time + peak memory ---
        torch.cuda.reset_peak_memory_stats()
        ms_tri = triton.testing.do_bench(lambda: flash_attention_forward(Q, K, V))
        mem_tri = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB

        # --- SDPA: time + peak memory ---
        torch.cuda.reset_peak_memory_stats()
        ms_cub = triton.testing.do_bench(lambda: torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True))
        mem_cub = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB

        # attention FLOPs: QK^T + PV, each 2*n*n*head_dim, times batch*heads
        flops_full = 4 * batch * heads * n * n * head_dim      # what Triton actually computes
        flops_causal = 2 * batch * heads * n * n * head_dim     # what SDPA actually computes (approx)

        tflops_tri = flops_full / (ms_tri * 1e-3) / 1e12
        tflops_cub = flops_causal / (ms_cub * 1e-3) / 1e12
        print(f"{n:>6} | {tflops_tri:>10.2f} | {tflops_cub:>10.2f} | "
              f"{tflops_tri / PEAK_TFLOPS * 100:>7.1f}% | "
              f"{tflops_cub / PEAK_TFLOPS * 100:>7.1f}% | "
              f"{mem_tri:>7.1f} | {mem_cub:>7.1f}")
        
# main
if __name__ == "__main__":
    output_run_fwd = run_fa_fwd(2,4,512,64)
    print(output_run_fwd)
    output_preprocess_kernel = test_preprocess_kernel(2,4,512,64)
    print(output_preprocess_kernel)
    # print("\n Benchmark testing \n")
    # benchmark(2,4,64)


# output 

# Before Autotune : 
# (mlenv) vraj@Vraj:/mnt/c/dev/projects/cpp-gpu-inference/5. flash-attention$ python flash_attention_fwd.py
# max abs diff: 0.00048828125
# mean abs diff: 0.00017583370208740234
# True

# After Autotune : 
# (mlenv) vraj@Vraj:/mnt/c/dev/projects/cpp-gpu-inference/5. flash-attention$ python flash_attention_fwd.py
# max abs diff: 0.00048828125
# mean abs diff: 0.00017392635345458984
# True


# # benchmark results 
#  Benchmark testing 

#   Size |     Triton |     cuBLAS |  Tri %pk |  cuB %pk |   Tri MB |   cuB MB
# ------------------------------------------------------------------------------------------
#    512 |      18.44 |      11.36 |    37.5% |    23.1% |   258.0 |   258.0
#   1024 |      30.13 |      21.03 |    61.3% |    42.8% |   260.0 |   260.0
#   2048 |      40.40 |      36.69 |    82.2% |    74.7% |   264.0 |   264.1
#   4096 |      46.19 |      52.89 |    94.0% |   107.6% |   272.0 |   272.1
#   8192 |      46.78 |      62.98 |    95.2% |   128.1% |   288.0 |   288.3