# softmax triton kernel

# basic steps for softmax : 
# 1. m = max(row)
# 2. s = row - m
# 3. e = exp(s)
# 4. out = e / sum(e)

# imports
import torch
import triton
import triton.language as tl
from triton.testing import do_bench

# device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# kernel
@triton.jit
def softmax_kernel(input_ptr, output_ptr, input_stride, output_stride, n_cols, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(axis=0)

    row_start = pid * input_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    offsets = row_start + col_offsets 

    out_offsets = pid * output_stride + col_offsets

    row = tl.load(input_ptr + offsets, mask=mask, other=float("-inf"))

    # 1. m - max(row)
    m = tl.max(row, axis=0)

    # 2. s = row - m
    s = row - m

    # 3. e - exp(s)
    e = tl.exp(s)

    # 4. out = e / sum(e)
    den = tl.sum(e, axis=0)
    out = e / den

    tl.store(output_ptr + out_offsets, out, mask=mask)


# function
def softmax(arr):
    output = torch.empty_like(arr)

    # making sure that the tensors are on the same device
    assert arr.is_cuda , "Triton requires CUDA tensors"

    # defining the launchpad grid
    n_rows, n_cols = arr.shape
    grid = (n_rows,)

    softmax_kernel[grid](
        arr,
        output,
        arr.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE= triton.next_power_of_2(n_cols) # this triton func makes sure to give it the next power of 2 as the block size. and here in this case, it's 1024
    )

# here, we have to use the stride() method because: 
# we have a 2d matrix and we need to jumps to next rows.
# now this 2d matrix is esentially 1d array internally. so in order to access it's index, we use the flat indenxing technique
# for ex, if we want the elements at row 5, and if we are at row 0 currently, then we just do 5 * 781 (here 781 is n_cols)
# therefore, the final flatindex is 5 * n_cols. 
# now this whole process is done by the method 'stride'. 

# in short terms ; A stride is: how many elements do I skip to advance by one along this dimension.

# stride(0) is row dimension and stride(1) is column dimension

# for our (1823, 781)
# arr.stride(0) == 781    # skip 781 floats to get to the next row
# arr.stride(1) == 1      # neighbors in a row are adjacent

# ------------------------------------------------------------------------------------------------

# detailed explaination on the arr.stride and output.stride 

# passing input and output strides separately, not one shared value.
# stride(0) = how many floats to skip to get from the start of one row
# to the start of the next.
#
# arr and output are two separate allocations, and the kernel only gets
# bare pointers - it can't inspect them to figure out how they're laid out.
# so each pointer needs its own stride to be usable.
#
# right now both are 781, because empty_like copies the layout too.
# but same shape does not mean same layout: a transposed or sliced view
# shares memory with the original and just reports different strides.
# hand this kernel a transposed input and arr.stride(0) becomes 1 while
# output.stride(0) stays 781 - so a kernel that assumed they matched
# would read down a column and write across a row. every value wrong,
# no error, no crash.

    return output


# main function
def run_softmax(size, atol=1e-3, rtol=1e-3, device=DEVICE):
    torch.manual_seed(42)
    arr = torch.randn(size, device=device)

    # print first few elements of array 
    print(arr[0:10])

    # output vars
    softmax_arr = softmax(arr)
    print("Softmax done on the given array")

    # print first few elements of softmax array 
    print(softmax_arr[0:10])

    torch_softmax = torch.softmax(arr, axis=1)
    
    print("Comparing both")
    torch.testing.assert_close(softmax_arr, torch_softmax, atol=atol, rtol=rtol)

    print("Passed!!")


def benchmark(n_rows=1823, widths=(128, 781, 1024, 2048, 4096), device=DEVICE):
    print(f"{'n_cols':>8} {'triton (ms)':>13} {'torch (ms)':>12} {'speedup':>9}")

    for n_cols in widths:
        arr = torch.randn((n_rows, n_cols), device=device)

        # correctness first - a fast wrong kernel is worthless
        torch.testing.assert_close(
            softmax(arr), torch.softmax(arr, axis=1), atol=1e-3, rtol=1e-3
        )

        tri_ms = do_bench(lambda: softmax(arr), warmup=25, rep=100, return_mode="mean")
        tor_ms = do_bench(lambda: torch.softmax(arr, axis=1), warmup=25, rep=100, return_mode="mean")

        print(f"{n_cols:>8} {tri_ms:>13.4f} {tor_ms:>12.4f} {tor_ms / tri_ms:>8.2f}x")


if __name__ == "__main__":

    run_softmax(size=(1823,781))
    print(torch.cuda.get_device_capability())
    benchmark()


# output

# tensor([[ 0.1940,  2.1614, -0.1721,  ..., -0.8353, -1.6443,  0.1315],
#         [-0.1797,  0.9744, -2.0136,  ...,  0.0543,  0.2508, -0.2370],
#         [ 1.1266,  1.5895,  0.1072,  ..., -0.4797,  1.4313,  1.7260],
#         ...,
#         [ 1.9695,  0.5996,  0.5583,  ..., -0.0692,  2.9254,  0.9274],
#         [-1.7301,  1.2562, -1.2144,  ..., -0.4262, -2.0327, -1.0277],
#         [-0.2987,  0.1946,  1.1640,  ..., -0.2081, -1.0439,  0.7082]],
#        device='cuda:0')
# Softmax done on the given array
# tensor([[0.0009, 0.0065, 0.0006,  ..., 0.0003, 0.0001, 0.0009],
#         [0.0007, 0.0021, 0.0001,  ..., 0.0008, 0.0010, 0.0006],
#         [0.0023, 0.0037, 0.0008,  ..., 0.0005, 0.0032, 0.0043],
#         ...,
#         [0.0053, 0.0013, 0.0013,  ..., 0.0007, 0.0137, 0.0019],
#         [0.0001, 0.0028, 0.0002,  ..., 0.0005, 0.0001, 0.0003],
#         [0.0006, 0.0009, 0.0025,  ..., 0.0006, 0.0003, 0.0016]],
#        device='cuda:0')
# Comparing both
# Passed!!
#   n_cols   triton (ms)   torch (ms)   speedup
#      128        0.0125       0.0114     0.91x
#      781        0.0227       0.0249     1.10x
#     1024        0.0262       0.0270     1.03x
#     2048        0.0463       0.0825     1.78x
#     4096        0.0912       0.1116     1.22x