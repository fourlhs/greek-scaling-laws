from tokenizers import Tokenizer

# Load the trained tokenizer
tokenizer = Tokenizer.from_file("greek_bpe_tokenizer.json")

# Greek test query
greek_query = "Τι είναι η τεχνητή νοημοσύνη;"

print(f"Original query: {greek_query}")
print()

# Encode
encoded = tokenizer.encode(greek_query)
print(f"Tokens: {encoded.tokens}")
print(f"IDs: {encoded.ids}")
print()

# Decode
decoded = tokenizer.decode(encoded.ids)
print(f"Decoded: {decoded}")