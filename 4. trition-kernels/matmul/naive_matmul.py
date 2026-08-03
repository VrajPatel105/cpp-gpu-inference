# naive matrix multiplication 

import triton
import triton.language as tl
import torch

# kernel here : 


def naive_matmul(A,B):
    M,K = A.shape
    K2,N = B.shape

    assert K == K2

    C = torch.zeroes((M,N), device=A.device, dtype=A.dtype)

    grid = (M*N,)

    naive_matmul_kernel[grid](
        A,B,C,
        M,N,K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )

    return C