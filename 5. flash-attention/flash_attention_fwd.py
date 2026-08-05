import torch
import triton
import triton.language as tl

# device 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# kernel
@triton.jit
def flash_attention_kernel():
    pass



def flash_attention_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    pass


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