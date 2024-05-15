from transformers import MarianTokenizer

# Example using a MarianMT tokenizer for English to German translation
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")

text = "This is a sample text to estimate the number of tokens."
encoded = tokenizer.encode(text)

print("Number of tokens:", len(encoded))
print("Tokenized output:", encoded)
