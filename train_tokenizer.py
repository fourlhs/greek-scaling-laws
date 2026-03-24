from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Sequence, NFD, Lowercase

def get_training_corpus(dataset):
    batch_size = 1000
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + 1000]["greek_translation"]

dataset = load_dataset("alexliap/tinystories-gr", split="train")

tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
tokenizer.normalizer = Sequence([NFD(), Lowercase()])
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=16000,
    special_tokens=["<|endoftext|>", "<|unk|>", "<|pad|>"],
    min_frequency=2
)

tokenizer.train_from_iterator(get_training_corpus(dataset), trainer=trainer)
tokenizer.save("greek_bpe_tokenizer.json")