import torch
import triton
import triton.language as tl

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
    seq_len, head_dim, 
    BLOCK_M : tl.constexpr,
    BLOCK_N : tl.constexpr
):
    pid_batch = tl.program_id(axis=0)
    pid_head = tl.program_id(axis=1)
    pid_m = tl.program_id(axis=2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M) 
    offs_d = 
    offs_q = pid_batch * stride_qb + pid_head * stride_qh + offs_m * stride_qs + offs_d * stride_qd



def flash_attention_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):

    BLOCK_M = 128
    BLOCK_N = 128
    
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
        BLOCK_M, BLOCK_N
    )


def run_fa_fwd(batch, heads, seq_len, head_dim):

    # initialize the Q, K and V tensors
    Q = torch.rand(batch, heads, seq_len, head_dim, device=DEVICE)
    K = torch.rand(batch, heads, seq_len, head_dim, device=DEVICE)
    V = torch.rand(batch, heads, seq_len, head_dim, device=DEVICE)

    # call the wrapper function
    flash_attention_forward(Q,K,V)


# main
if __name__ == "__main__":
    run_fa_fwd(2,4,512,64)