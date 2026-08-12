import torch
import torch.nn as nn
import sys
sys.path.append("/mnt/c/dev/projects/cpp-gpu-inference/en-de-transformer")
from model import build_transformer
from quantized_linear import QuantizedLinear
from config import configurations
from tokenizer import Tokenizer
from train import translate, load_data
import bitsandbytes as bnb
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)

# 1. Load calibration data
calibration_data = torch.load('/mnt/c/dev/projects/cpp-gpu-inference/6. Quantization/llm.int8()/calibration_data.pt')
outlier_dict = calibration_data['outlier_dict']

# 2. Load  trained model and the checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_transformer(configurations)
checkpoint = torch.load('/mnt/c/dev/projects/cpp-gpu-inference/en-de-transformer/transformer_en_de.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()


def quantize_model(model, outlier_dict):
    for key in outlier_dict:
        parts = key.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        last_attr = parts[-1]
        old_layer = getattr(parent, last_attr)
        new_layer = QuantizedLinear(old_layer.weight, old_layer.bias, outlier_dict[key])
        setattr(parent, last_attr, new_layer)

    return model


with torch.no_grad():
    quantized_model = quantize_model(model, outlier_dict)
    torch.save(quantized_model.state_dict(), 'quantized_transformer_en_de.pt')

def translate(model, sentence, eng_tok, de_tok, device, max_len):
    model.eval()
    model.to(device)
    with torch.no_grad():
        
        tokenized_sentence = eng_tok.encode_sentence(sentence, add_sos=True, add_eos=True)
        assert len(tokenized_sentence) <= max_len, f"too long: {len(tokenized_sentence)} tokens"
        pad_count = max_len - len(tokenized_sentence)
        tokenized_sentence = tokenized_sentence + [eng_tok.PAD_ID] * pad_count

        encoder_input = torch.tensor(tokenized_sentence, dtype=torch.long).unsqueeze(0).to(device)
        src_mask = (encoder_input != eng_tok.PAD_ID).unsqueeze(1).unsqueeze(1).int()

        # KV CACHE: run encoder once explicitly instead of through model.forward()
        src = model.src_pe(model.src_embed(encoder_input))
        for block in model.encoder_blocks:
            src = block(src, src_mask)
        enc_output = src

        # KV CACHE: initialize empty caches per decoder layer
        num_layers = len(model.decoder_blocks)
        sa_caches = [None] * num_layers
        ca_caches = [None] * num_layers

        next_token_id = de_tok.SOS_ID
        generated_ids = [next_token_id]

        for _ in range(max_len):
            # KV CACHE: embed only the NEW token, not the full sequence
            token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)

            # KV CACHE: add positional encoding for THIS position only
            pos = len(generated_ids) - 1
            tgt = model.tgt_embed(token_tensor)
            tgt = tgt + model.tgt_pe.pe[:, pos:pos+1, :]

            # KV CACHE: no causal mask needed — Q is length 1, all cached K are past tokens
            tgt_mask = None

            # KV CACHE: run decoder blocks, threading caches through each layer
            for i, block in enumerate(model.decoder_blocks):
                tgt, sa_caches[i], ca_caches[i] = block(
                    tgt, enc_output, src_mask, tgt_mask,
                    sa_cache=sa_caches[i],
                    ca_cache=ca_caches[i]
                )

            logits = model.projection_layer(tgt)
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()

            generated_ids.append(next_token_id)

            if next_token_id == de_tok.EOS_ID:
                break

        ids = generated_ids[1:]  # drop SOS
        if ids and ids[-1] == de_tok.EOS_ID:
            ids = ids[:-1]  # drop EOS if present

        return de_tok.decode_sentence(ids)
    

def replace_with_bnb_linear(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            new_layer = bnb.nn.Linear8bitLt(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                has_fp16_weights=False,   # store weights in int8, not fp16
                threshold=6.0             # same outlier threshold concept as LLM.int8()
            )
            new_layer.weight = bnb.nn.Int8Params(
                child.weight.data.clone(), requires_grad=False, has_fp16_weights=False
            )
            if child.bias is not None:
                new_layer.bias = nn.Parameter(child.bias.data.clone())
            setattr(module, name, new_layer)
        else:
            replace_with_bnb_linear(child)  # recurse into submodules

# load bnb model
bnb_model = build_transformer(configurations).to(device)
bnb_checkpoint = torch.load(
    '/mnt/c/dev/projects/cpp-gpu-inference/en-de-transformer/transformer_en_de.pt',
    map_location=device
)
bnb_model.load_state_dict(bnb_checkpoint['model_state_dict'])

replace_with_bnb_linear(bnb_model)
bnb_model = bnb_model.to(device)  # bnb layers quantize weights to int8 once moved to CUDA
bnb_model.eval()


def main():

    english, german = load_data(configurations['path'])
    english_tokenizer = Tokenizer()
    english_tokenizer.build_vocab(english)
    german_tokenizer = Tokenizer()
    german_tokenizer.build_vocab(german)

    # Build a fresh, UNQUANTIZED model for comparison
    original_model = build_transformer(configurations).to(device)
    original_checkpoint = torch.load(
        '/mnt/c/dev/projects/cpp-gpu-inference/en-de-transformer/transformer_en_de.pt',
        map_location=device
    )
    original_model.load_state_dict(original_checkpoint['model_state_dict'])
    original_model.eval()

    sentences = [
        "I am hungry.",
        "Hello.",
        "I am tired.",
        "The book is on the table.",
        "She is my friend.",
        "What time is it?",
        "I am happy.",
        "He is at home.",
        "The dog is sleeping.",
        "This is my car.",
        "I like coffee.",
        "They are students.",
        "Where is the bathroom?",
        "It is raining today.",
        "Please open the door.",
    ]

    max_len = configurations['max_len']
    for s in sentences:
        print(f"EN: {s}")
        original_translation = quantized_translation = bnb_translation = None

        try:
            original_translation = translate(original_model, s, english_tokenizer, german_tokenizer, device, max_len=max_len)
            print(f"  ORIGINAL : {original_translation}")
        except AssertionError as e:
            print(f"  ORIGINAL : skipped ({e})")

        try:
            quantized_translation = translate(quantized_model, s, english_tokenizer, german_tokenizer, device, max_len=max_len)
            print(f"  QUANTIZED: {quantized_translation}")
        except AssertionError as e:
            print(f"  QUANTIZED: skipped ({e})")

        try:
            bnb_translation = translate(bnb_model, s, english_tokenizer, german_tokenizer, device, max_len=max_len)
            print(f"  BNB: {bnb_translation}")
        except AssertionError as e:
            print(f"  BNB: skipped ({e})")

        match = "MATCH" if original_translation == quantized_translation == bnb_translation else "DIFFER"
        print(f"  -> {match}\n")


if __name__ == "__main__":
    main()

"""
# output 
EN: I am hungry.
  ORIGINAL : ich habe hunger
  QUANTIZED: ich habe hunger
  BNB: ich habe hunger
  -> MATCH

EN: Hello.
  ORIGINAL : hallo
  QUANTIZED: hallo
  BNB: hallo
  -> MATCH

EN: I am tired.
  ORIGINAL : ich bin müde
  QUANTIZED: ich bin müde
  BNB: ich bin müde
  -> MATCH

EN: The book is on the table.
  ORIGINAL : das buch ist auf dem tisch
  QUANTIZED: das buch ist auf dem tisch
  BNB: das buch ist auf dem tisch
  -> MATCH

EN: She is my friend.
  ORIGINAL : sie ist mein freund
  QUANTIZED: sie ist mein freund
  BNB: sie ist mein freund
  -> MATCH

EN: What time is it?
  ORIGINAL : was ist es zeit
  QUANTIZED: was ist es zeit
  BNB: was ist es zeit
  -> MATCH

EN: I am happy.
  ORIGINAL : ich bin glücklich
  QUANTIZED: ich bin glücklich
  BNB: ich bin glücklich
  -> MATCH

EN: He is at home.
  ORIGINAL : er ist zu hause
  QUANTIZED: er ist zu hause
  BNB: er ist zu hause
  -> MATCH

EN: The dog is sleeping.
  ORIGINAL : der hund schläft
  QUANTIZED: der hund schläft
  BNB: der hund schläft
  -> MATCH

EN: This is my car.
  ORIGINAL : das ist mein auto
  QUANTIZED: das ist mein auto
  BNB: das ist mein auto
  -> MATCH

EN: I like coffee.
  ORIGINAL : ich mag kaffee
  QUANTIZED: ich mag kaffee
  BNB: ich mag kaffee
  -> MATCH

EN: They are students.
  ORIGINAL : sie sind studenten
  QUANTIZED: sie sind studenten
  BNB: sie sind studenten
  -> MATCH

EN: Where is the bathroom?
  ORIGINAL : wo ist der weg
  QUANTIZED: wo ist der weg
  BNB: wo ist der weg
  -> MATCH

EN: It is raining today.
  ORIGINAL : es regnet heute
  QUANTIZED: es regnet heute
  BNB: es regnet heute
  -> MATCH

EN: Please open the door.
  ORIGINAL : bitte öffnen sie die tür
  QUANTIZED: bitte öffnen sie die tür
  BNB: bitte öffnen sie die tür
  -> MATCH

"""