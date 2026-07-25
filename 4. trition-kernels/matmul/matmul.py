import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):

    # 1. which tile am I? two program ids
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    # 2. offs_m, offs_n, offs_k  — three arange vectors
    offs_m = pid_x + BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_y + BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 3. build 2D pointer blocks for A and B using [:, None] / [None, :]
    row_a = offs_m[:, None] * stride_am
    col_a = offs_k[None, :] * stride_ak

    a_ptrs = row_a * stride_am + col_a * stride_ak
    # 4. accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # 5. for k in range(0, tl.cdiv(K, BLOCK_K)):
    #        load a tile, load b tile
    #        accumulator += tl.dot(a, b)
    #        advance both pointers along K
    # 6. store accumulator to C with a 2D mask

    pass


def matmul(x,y, M, N, K):

    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    assert x.is_cuda , "Triton requires CUDA tensors"

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_kernel[grid](
        x, y, output,
        M, N, K,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
    )
    return output



def run_matmul_kernel(M, N, K, atol=1e-3, rtol=1e-3, device=DEVICE):
    torch.manual_seed(42)
    x = torch.randn((M, K), device=device)
    y = torch.randn((K, N), device=device)

    # define output vars
    z_tri = matmul(x, y, M, N, K)
    z_ref = x @ y 

    # compare
    torch.testing.assert_close(z_tri, z_ref, atol=atol, rtol=rtol)
    print("Passed!!")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("No CUDA GPU available")
    run_matmul_kernel(512, 384, 256)