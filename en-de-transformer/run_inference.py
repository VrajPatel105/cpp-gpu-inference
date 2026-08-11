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
    checkpoint = torch.load('/mnt/c/dev/projects/cpp-gpu-inference/en-de-transformer/transformer_en_de.pt', map_location=device)
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
            translation = translate(model, s, english_tokenizer, german_tokenizer, device, max_len=max_len)
            print(f"EN: {s}")
            print(f"DE: {translation}\n")
        except AssertionError as e:
            print(f"EN: {s}")
            print(f"   skipped: {e}\n")


if __name__ == "__main__":
    main()


# OUTPUT FROM THE MODEL WHEN RUNNING THE ABOVE CODE



# EN: I am hungry.
# DE: ich bin hungrig

# EN: Hello.
# DE: ist

# EN: I am tired.
# DE: ich bin müde

# EN: The book is on the table.
# DE: der auf dem schreibtisch

# EN: She is my friend.
# DE: sie ist dich

# EN: What time is it?
# DE: was ist

# EN: I am happy.
# DE: ich bin glücklich

# EN: He is at home.
# DE: er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist er ist

# EN: The dog is sleeping.
# DE: der nähe

# EN: This is my car.
# DE: das auto

# EN: I like coffee.
# DE: ich habe

# EN: They are students.
# DE: sie sind sie sind sie sind sie sind sie

# EN: Where is the bathroom?
# DE: wo ist das klo

# EN: It is raining today.
# DE: es regnet

# EN: Please open the door.
# DE: bitte die tür

"""
Retraining results : (drastic improvement)

Checkpoint Loaded 

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