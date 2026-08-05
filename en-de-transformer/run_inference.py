import torch
from model import build_transformer
from tokenizer import Tokenizer
from config import configurations
from train import translate, load_data


def main():

    english, german = load_data(configurations['path'])
    english_tokenizer = Tokenizer()
    english_tokenizer.build_vocab(english)
    german_tokenizer = Tokenizer()
    german_tokenizer.build_vocab(german)

    # building the model skeleton and then loading the trained weights if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_transformer(configurations).to(device)

    # loading the checkpoint
    checkpoint = torch.load('transformer_en_de.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Checkpoint Loaded \n")

    # test sentences
    sentences = [
        "I am hungry.",
        "Hello.",
        "I am tired.",
        "The book is on the table.",
        "She is my friend.",
        "What time is it?",
    ]

    max_len = configurations['max_len']
    for s in sentences:
        try:
            translation = translate(model, s, english_tokenizer, german_tokenizer, device, max_len=max_len)
            print(f"EN: {s}")
            print(f"DE: {translation}\n")
        except AssertionError as e:
            print(f"EN: {s}")
            print(f"   skipped: {e}\n")


if __name__ == "__main__":
    main()


# OUTPUT FROM THE MODEL WHEN RUNNING THE ABOVE CODE

"""
EN: I am hungry.
DE: ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin ich bin

EN: Hello.
DE: das

EN: I am tired.
DE: ich bin

EN: The book is on the table.
DE: die

EN: She is my friend.
DE: sie ist mein

EN: What time is it?
DE: es das was zeit

"""