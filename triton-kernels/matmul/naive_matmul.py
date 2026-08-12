# naive matrix multiplication 

import triton
import triton.language as tl
import torch

# kernel here : 
@triton.jit
def naive_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn
):

    pid = tl.program_id(axis=0)

    m = pid // N
    n = pid % N

    if (m >= M) or (n >= N):
        return

    accumulator = 0.0
    for k in range(0,K):
        a_val = tl.load(A_ptr + m * stride_am + k * stride_ak)
        b_val = tl.load(B_ptr + n * stride_bn + k * stride_bk)
        accumulator += a_val * b_val

    tl.store(C_ptr + m * stride_cm + n * stride_cn, accumulator)



def naive_matmul(A,B):
    M,K = A.shape
    K2,N = B.shape

    assert K == K2

    C = torch.zeros((M,N), device=A.device, dtype=A.dtype)

    grid = (M*N,)

    naive_matmul_kernel[grid](
        A,B,C,
        M,N,K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )

    return C

A = torch.randn(6,8, device="cuda")
B = torch.randn(8,4, device="cuda")

print(naive_matmul(A,B))

# output 

# tensor([[-1.1018, -1.2140, -6.2719, -2.3341],
#         [ 3.0004,  1.7936,  1.9925, -2.5540],
#         [-0.6565, -0.2868, -5.0976, -1.4120],
#         [ 4.3444,  1.0266,  4.5845, -0.1111],
#         [ 1.3867,  2.2309, -4.8464,  2.2810],
#         [ 1.7592,  0.0587, -0.7374, -0.3038]], device='cuda:0')