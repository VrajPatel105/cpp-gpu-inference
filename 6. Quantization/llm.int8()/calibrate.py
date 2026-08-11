import torch
import torch.nn as nn
import sys
sys.path.append("/mnt/c/dev/projects/cpp-gpu-inference/en-de-transformer")

from train import build_transformer, val_loader
from config import configurations
from utils import make_masks 

model = build_transformer(configurations)
model_checkpoint = torch.load("/mnt/c/dev/projects/cpp-gpu-inference/en-de-transformer/transformer_en_de.pt")
model.load_state_dict(model_checkpoint['model_state_dict'])
pad_id = 0

running_max = {}

def make_hook(name):
    def hook_func(module, input, output):
        x = input[0] # the input is just (tensor,) but pytorch func requires this to be a tuple.
        # lets get the named vars just for more simplicity
        batch = x.shape[0]
        seq_len = x.shape[1]
        hidden_dim = x.shape[2]
        # first reshape 
        x = x.reshape(batch*seq_len, hidden_dim)
        # now convert the values in hidden_dim to abs (meaning to simply apply modulus)
        x = x.abs()
        # now get the max on hidden_dim col (channel)
        x = x.max(dim=0).values

        # this finally means that we have per channel max for that tensor (channel aka hidden_dim)

        # now updating the running max
        if name not in running_max:
            running_max[name] = x 
        else:
            running_max[name] = torch.maximum(running_max[name], x)
    return hook_func

for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        module.register_forward_hook(make_hook(name))

def calibrate(model, loader, device, pad_id):
    model.eval()
    with torch.no_grad():
        cnt = 0 # counter for calibrating -> batch size
        for batch in loader:
            if cnt >= 32:
                break 
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            label = batch["label"].to(device)

            src_mask, tgt_mask = make_masks(encoder_input, decoder_input, pad_id, device)

            output = model(encoder_input, decoder_input, src_mask, tgt_mask)

            cnt += 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
calibrate(model, val_loader, device, pad_id)


print(f"Calibrated {len(running_max)} layers")
for name, vals in list(running_max.items())[:3]:
    print(name, vals.shape, vals.max().item())

# output for aboev loop
# (mlenv) vraj@Vraj:/mnt/c/dev/projects/cpp-gpu-inference/6. Quantization/llm.int8()$ python calibrate.py
# Calibrated 97 layers
# encoder_blocks.0.multi_head_attention.W_q torch.Size([512]) 101.70893859863281
# encoder_blocks.0.multi_head_attention.W_k torch.Size([512]) 101.70893859863281
# encoder_blocks.0.multi_head_attention.W_v torch.Size([512]) 101.70893859863281
# (mlenv) vraj@Vraj:/mnt/c/dev/projects/cpp-gpu-inference/6. Quantization/llm.int8()$ 