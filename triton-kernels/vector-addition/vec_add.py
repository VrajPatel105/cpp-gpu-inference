# summary that i wrote on my own (it's a walkthrough)

# walking through element 2049:
#
# main calls add(), which launches add_kernel.
#
# pid = 2 handles it, because pid = 1 only covers 1024 to 2047.
# block_start = 2 * BLOCK_SIZE = 2 * 1024 = 2048
# offsets = block_start + tl.arange(0, BLOCK_SIZE)
#         = 2048 + [0, 1, ... 1023]
#         = [2048, 2049, ... 3071]      (3072 belongs to pid = 3, arange excludes the end)
# so element 2049 is lane 1 inside this block.
#
# mask = offsets < n_elements -> everything is under 4096 here, so it's all True.
# mask is a bool vector, one True/False per lane.
#
# x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
# x_ptr + offsets gives 1024 addresses, but tl.load goes and reads them,
# so x comes back as the actual 1024 values sitting in registers. not addresses.
# same for y.
#
# x + y adds them elementwise, all 1024 lanes at once, still in registers.
#
# tl.store(output_ptr + offsets, output, mask=mask)
# writes the results back to DRAM at the same offsets we loaded from,
# so index 2048 to 3071 of output now holds the final values.



# vector addition triton kernel. 
import torch
import triton
import triton.language as tl

# select device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# defining the actual kernel
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # load data from dram
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    output = x + y

    # writing the data back to dram
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x, y):
    # pre allocate the output -> making the z tensor in the mem in existance
    output = torch.empty_like(x)

    # making sure that the tensors are on the same device
    assert x.device == y.device
    assert x.is_cuda and y.is_cuda, "Triton requires CUDA tensors"

    # defining our launch grid
    n_elements = output.numel()  # this gives the total number of numbers in that tensor
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    add_kernel[grid](
        x,
        y,
        output,
        n_elements,
        BLOCK_SIZE=1024
    )

    return output


def run_add_kernel(size, atol=1e-3, rtol=1e-3, device=DEVICE):
    torch.manual_seed(42)
    x = torch.randn(size, device=device)
    y = torch.randn(size, device=device)

    # define output vars
    z_tri = add(x, y)
    z_ref = x + y  # this is for us to compare the z_tri (triton output) with this reference 'pytorch' calculations

    # compare
    torch.testing.assert_close(z_tri, z_ref, atol=atol, rtol=rtol)
    print("Passed!!")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("No CUDA GPU available")
    run_add_kernel(size=4096)

# output 
# Passed!!