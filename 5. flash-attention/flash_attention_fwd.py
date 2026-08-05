import torch
import triton
import triton.language as tl
import math

# device 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# kernel
@triton.jit
def flash_attention_kernel(
    Q, K, V, O,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
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

        # compute the running vars 
        # 1. rowmax 
        rowmax = tl.max(S, axis=1)
        # 2. max
        m_new = tl.maximum(m, rowmax)
        # 3. rowsum
        tilde_p = tl.exp(S - m_new)
        rowsum = tl.sum(tilde_p, axis=1) 
        l_new = tl.exp(m - m_new) * l + rowsum

        # now o
        rescale_factor = tl.exp(m - m_new)
        old_c = rescale_factor[:, None] * o # this broadcasts the per row scalar across head_dim
        new_c = tl.dot(tilde_p, v_tile)
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


def flash_attention_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):

    BLOCK_M = 64
    BLOCK_N = 64
    
    # extracting the size for q k v tensors
    batch, num_heads, seq_len, head_dim = Q.shape
    assert K.shape == Q.shape and V.shape == Q.shape, "Q, K, V shape mismatch"

    assert Q.is_cuda and K.is_cuda and V.is_cuda , "Not CUDA Tensors -_-"

    # output tensor
    O = torch.empty(batch, num_heads, seq_len, head_dim, device=DEVICE)

    # define the launchpad grid
    grid = (batch, num_heads, triton.cdiv(seq_len, BLOCK_M))

    flash_attention_kernel[grid](
        Q, K, V, O,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        seq_len, head_dim,
        BLOCK_M, BLOCK_N,
    )

    return O


def run_fa_fwd(batch, heads, seq_len, head_dim):

    torch.manual_seed(42)

    # initialize the Q, K and V tensors
    Q = torch.rand(batch, heads, seq_len, head_dim, device=DEVICE)
    K = torch.rand(batch, heads, seq_len, head_dim, device=DEVICE)
    V = torch.rand(batch, heads, seq_len, head_dim, device=DEVICE)

    # call the wrapper function
    output_flash_attention = flash_attention_forward(Q,K,V)
    output_torch = torch.softmax(Q@K.transpose(-2,-1)/math.sqrt(head_dim), dim=-1) @ V

    result = torch.allclose(output_flash_attention, output_torch, atol=1e-2, rtol=1e-3)
    diff = (output_flash_attention - output_torch).abs()
    print("max abs diff:", diff.max().item())
    print("mean abs diff:", diff.mean().item())

    return result

# main
if __name__ == "__main__":
    outupt = run_fa_fwd(2,4,512,64)
    print(outupt)


# output 
# (mlenv) vraj@Vraj:/mnt/c/dev/projects/cpp-gpu-inference/5. flash-attention$ python flash_attention_fwd.py
# max abs diff: 0.010206371545791626
# mean abs diff: 0.0019075826276093721
# True