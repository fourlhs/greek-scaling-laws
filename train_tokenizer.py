from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

def get_training_corpus(dataset):
    batch_size = 1000
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + 1000]["greek_translation"]

dataset = load_dataset("alexliap/tinystories-gr", split="train")

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=16000,
    special_tokens=["<|endoftext|>", "<|pad|>"],
    initial_alphabet=ByteLevel.alphabet()
)

tokenizer.train_from_iterator(get_training_corpus(dataset), trainer=trainer)
tokenizer.save("greek_bpe_tokenizer.json")