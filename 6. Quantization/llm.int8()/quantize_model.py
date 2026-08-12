import torch
import torch.nn as nn
import sys
sys.path.append("/mnt/c/dev/projects/cpp-gpu-inference/en-de-transformer")
from model import build_transformer
from quantized_linear import QuantizedLinear
from config import configurations
from tokenizer import Tokenizer
from train import translate, load_data

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



def main():

    english, german = load_data(configurations['path'])
    english_tokenizer = Tokenizer()
    english_tokenizer.build_vocab(english)
    german_tokenizer = Tokenizer()
    german_tokenizer.build_vocab(german)

    # test sentences
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
        try:
            translation = translate(quantized_model, s, english_tokenizer, german_tokenizer, device, max_len=max_len)
            print(f"EN: {s}")
            print(f"DE: {translation}\n")
        except AssertionError as e:
            print(f"EN: {s}")
            print(f"   skipped: {e}\n")


if __name__ == "__main__":
    main()

"""
# output 
EN: I am hungry.
DE: ich habe hunger

EN: Hello.
DE: hallo

EN: I am tired.
DE: ich bin müde

EN: The book is on the table.
DE: das buch ist auf dem tisch

EN: She is my friend.
DE: sie ist mein freund

EN: What time is it?
DE: was ist es zeit

EN: I am happy.
DE: ich bin glücklich

EN: He is at home.
DE: er ist zu hause

EN: The dog is sleeping.
DE: der hund schläft

EN: This is my car.
DE: das ist mein auto

EN: I like coffee.
DE: ich mag kaffee

EN: They are students.
DE: sie sind studenten

EN: Where is the bathroom?
DE: wo ist der weg

EN: It is raining today.
DE: es regnet heute

EN: Please open the door.
DE: bitte öffnen sie die tür

"""