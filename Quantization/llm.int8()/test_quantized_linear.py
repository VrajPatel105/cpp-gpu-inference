import torch
import sys
sys.path.append("/mnt/c/dev/projects/cpp-gpu-inference/en-de-transformer")

from train import build_transformer
from config import configurations
from quantized_linear import QuantizedLinear

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  load model + calibration data 
model = build_transformer(configurations)
checkpoint = torch.load("/mnt/c/dev/projects/cpp-gpu-inference/en-de-transformer/transformer_en_de.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

calibration_data = torch.load('calibration_data.pt')
outlier_dict = calibration_data['outlier_dict']

#  pick one real layer to test 
layer_name = "encoder_blocks.0.multi_head_attention.W_q"
original_layer = model.encoder_blocks[0].multi_head_attention.W_q  # adjust attribute path to match your model

#  build the quantized version from that layer's real weights 
outlier_idx = outlier_dict[layer_name]
q_layer = QuantizedLinear(original_layer.weight, original_layer.bias, outlier_idx).to(device)

#  test input 
x = torch.randn(4, 512, device=device)  # [batch, in_features] — match d_model

# compare outputs 
with torch.no_grad():
    y_original = original_layer(x)
    y_quantized = q_layer(x)

print("max abs diff:", (y_original - y_quantized).abs().max().item())
print("mean abs diff:", (y_original - y_quantized).abs().mean().item())
print("original output sample:", y_original[0, :5])
print("quantized output sample:", y_quantized[0, :5])

# ouput 
# max abs diff: 0.018088340759277344
# mean abs diff: 0.0037627005949616432
# original output sample: tensor([-0.1902,  0.4178, -0.0069,  0.0248,  0.5384], device='cuda:0')
# quantized output sample: tensor([-0.1891,  0.4159, -0.0019,  0.0228,  0.5321], device='cuda:0')