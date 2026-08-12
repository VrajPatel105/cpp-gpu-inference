import torch
import time

N = 1024
A = torch.ones(N, N, device='cuda', dtype=torch.float32)
B = torch.ones(N, N, device='cuda', dtype=torch.float32)

# warmup — first run is always slow, ignore it
_ = torch.matmul(A, B)
torch.cuda.synchronize()

# time it
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
C = torch.matmul(A, B)
end.record()

torch.cuda.synchronize()
print(f"torch.matmul: {start.elapsed_time(end):.3f} ms")