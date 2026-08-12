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


# print(f"Calibrated {len(running_max)} layers")
# for name, vals in list(running_max.items())[:3]:
#     print(name, vals.shape, vals.max().item())

# output for aboev loop
# (mlenv) vraj@Vraj:/mnt/c/dev/projects/cpp-gpu-inference/6. Quantization/llm.int8()$ python calibrate.py
# Calibrated 97 layers
# encoder_blocks.0.multi_head_attention.W_q torch.Size([512]) 101.70893859863281
# encoder_blocks.0.multi_head_attention.W_k torch.Size([512]) 101.70893859863281
# encoder_blocks.0.multi_head_attention.W_v torch.Size([512]) 101.70893859863281
# (mlenv) vraj@Vraj:/mnt/c/dev/projects/cpp-gpu-inference/6. Quantization/llm.int8()$ 


# now we get the final block here which is to find out which channels from the layers
# that are stored in the running_max dict have value above a threshold value
# we will use alpha = 6 as implemented in the llm.int8() paper.

# iterating through the dict

outlier_dict = {}

# for key, value in list(running_max.items()):
#     mask = (value > 6.0).to(device)
#     outlier_indices = torch.arange(value.numel(), device=mask.device)[mask]
#     outlier_dict[key] = outlier_indices

# key = "encoder_blocks.0.multi_head_attention.W_k"
# print("\n" + key )
# print(running_max[key].min().item())
# print(running_max[key].max().item())
# print(running_max[key].mean().item())

# key = "encoder_blocks.1.multi_head_attention.W_k"
# print("\n" + key )
# print(running_max[key].min().item())
# print(running_max[key].max().item())
# print(running_max[key].mean().item())


# findings  

# encoder_blocks.0.multi_head_attention.W_k
# 58.67816925048828
# 101.70893859863281
# 76.09907531738281

# encoder_blocks.1.multi_head_attention.W_k
# 2.6325955390930176
# 4.541968822479248
# 3.359626293182373

# upon priting the first layer vector in outlier_indices dict, i found out that all the indices from 0 to 511 were getting flagged as outlier
# this was because all the values in that block were higher than the flag, which is alpha = 6 here (we took this from teh paper)
# but upon priting the values for the block1, the values are well within the range but then no channels are getting flagged in this blcok since there's no value greater than the alpha 
# as you can notice here, this means, we will have to implement dynamic alpha value per block rather than a standard global value

# here based on stats, i will be using this value : alpha = mean + 3*std -> we can easily term this value as outlier if it's well greater than 3 times it's std + mean

# modified new loop with per block alpha

for key, value in list(running_max.items()):
    alpha = torch.mean(value) + 3 * torch.std(value)
    mask = (value > alpha).to(device)
    outlier_indices = torch.arange(value.numel(), device=mask.device)[mask]
    outlier_dict[key] = outlier_indices

torch.save({
    'running_max': running_max,
    'outlier_dict': outlier_dict,
}, 'calibration_data.pt')

# for name, vals in list(outlier_dict.items())[82:85]:
#     print(name, vals)
# output for thsi loop
# decoder_blocks.4.cross_attention.W_v tensor([127, 318, 419, 475, 501], device='cuda:0')
# decoder_blocks.4.cross_attention.W_o tensor([127, 219, 461, 478], device='cuda:0')
# decoder_blocks.4.feed_forward.linear1 tensor([ 24, 131, 161, 330, 507], device='cuda:0')