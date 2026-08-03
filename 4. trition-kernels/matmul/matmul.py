import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
    reset_to_zero=None,
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):

    # 1. which tile am I? two program ids
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    # 2. offs_m, offs_n, offs_k  — three arange vectors
    offs_m = pid_x * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_y * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    # 4. accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] + k * BLOCK_K < K)
        mask_b = (offs_k[:, None] + k * BLOCK_K < K) & (offs_n[None, :] < N)

        a_tile = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b_tile = tl.load(b_ptrs, mask=mask_b, other=0.0)
        accumulator += tl.dot(a_tile, b_tile, allow_tf32=False)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=mask_c)

def matmul(x,y, M, N, K):

    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    assert x.is_cuda and y.is_cuda , "Triton requires CUDA tensors"

    grid = lambda meta: (
    triton.cdiv(M, meta['BLOCK_M']),
    triton.cdiv(N, meta['BLOCK_N']),
    )

    matmul_kernel[grid](
        x, y, output,
        M, N, K,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        output.stride(0), output.stride(1),
    )
    return output



def run_matmul_kernel(M, N, K, atol=1e-3, rtol=1e-3, device=DEVICE):
    torch.manual_seed(42)
    x = torch.randn((M, K), device=device)
    y = torch.randn((K, N), device=device)

    # define output vars
    z_tri = matmul(x, y, M, N, K)
    z_ref = x @ y

    print("Triton output:\n", z_tri)
    print("Torch output:\n", z_ref)
    print("Max abs diff:", (z_tri - z_ref).abs().max().item())

    torch.testing.assert_close(z_tri, z_ref, atol=1e-3, rtol=1e-3)
    print("Passed!!")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("No CUDA GPU available")
    run_matmul_kernel(512, 384, 256)

# output 
# Triton output:
#  tensor([[ -4.9429,   3.4138,   7.3413,  ...,   2.3503,  -9.6635,  11.9366],
#         [ 11.4245, -29.4036, -14.9556,  ...,  -6.5273,  10.0260, -16.1060],
#         [-30.5582,  34.0547,   2.0966,  ..., -14.5164,  23.3501,  14.7303],
#         ...,
#         [ 25.5702, -13.0629, -11.8159,  ...,   8.3727,   6.8580,  -9.2806],
#         [ 32.9803,  -3.7687, -29.9373,  ...,  26.2881,   8.8766,   2.8369],
#         [-20.9997,   4.4042, -10.1525,  ..., -25.7613, -10.0026, -44.1913]],
#        device='cuda:0')
# Torch output:
#  tensor([[ -4.9429,   3.4138,   7.3413,  ...,   2.3503,  -9.6635,  11.9366],
#         [ 11.4245, -29.4036, -14.9556,  ...,  -6.5273,  10.0260, -16.1060],
#         [-30.5582,  34.0547,   2.0966,  ..., -14.5164,  23.3501,  14.7303],
#         ...,
#         [ 25.5702, -13.0629, -11.8158,  ...,   8.3727,   6.8580,  -9.2806],
#         [ 32.9803,  -3.7687, -29.9373,  ...,  26.2881,   8.8766,   2.8369],
#         [-20.9997,   4.4042, -10.1525,  ..., -25.7613, -10.0026, -44.1913]],
#        device='cuda:0')
# Max abs diff: 4.57763671875e-05
# Passed!!