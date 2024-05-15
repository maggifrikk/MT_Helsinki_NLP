from retry import retry
from transformers import MarianMTModel, MarianTokenizer
import os
import sys
from time import time

# Specify your custom cache directory
cache_dir = os.path.expanduser('~/.cache/huggingface/transformers')

@retry(ConnectionError, tries=5, delay=2)
def translate(text, model_name):
    # Load the tokenizer and model
    tokenizer = MarianTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = MarianMTModel.from_pretrained(model_name, cache_dir=cache_dir)

    # Tokenize the text
    batch = tokenizer([text], return_tensors="pt", padding=True)

    # Generate translation outputs
    translated = model.generate(**batch)
    translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    
    return translated_text

def split_text(text):
    return text.split('\n')

def adjust_sentences(sentences, file_path):
    if not os.path.exists(file_path):
        return sentences
    f = open(file_path).read().splitlines()
    return sentences[len(f):]

def safe_translate(sentence, model_name):
    try:
        return translate(sentence, model_name)
    except IndexError as e:
        print(f"Error translating sentence: {e}")
        sentence_words = sentence.split()
        midpoint = len(sentence_words) // 2
        part1 = " ".join(sentence_words[:midpoint])
        part2 = " ".join(sentence_words[midpoint:])
        translated_part1 = translate(part1, model_name)
        translated_part2 = translate(part2, model_name)
        return translated_part1 + " " + translated_part2

s = time()
# Example usage
model_names = ["Helsinki-NLP/opus-mt-en-de", "Helsinki-NLP/opus-mt-de-en",
               "Helsinki-NLP/opus-mt-en-uk", "Helsinki-NLP/opus-mt-uk-en"]
            #    "Helsinki-NLP/opus-mt-en-zh", "Helsinki-NLP/opus-mt-zh-en",
            #    "Helsinki-NLP/opus-mt-en-he", "Helsinki-NLP/opus-mt-he-en",
            #    "Helsinki-NLP/opus-mt-cs-uk"]

# Directory containing source text files
source_dir = 'source_files'
output_dir = 'output_files'

for model_name in model_names:
    print(f'Translating using model: {model_name}')
    language_part = model_name[-5:]
    input_text_files = [filename for filename in os.listdir(source_dir) if language_part in filename]

    for file_name in input_text_files:
        print(f'Translating file: {file_name}')
        file_path = os.path.join(source_dir, file_name)
        output_file_path = os.path.join(output_dir, file_name + "_translation")

        with open(file_path, 'r') as file:
            input_text = file.read()
        
        sentences = split_text(input_text)
        print(sentences[-5:])
        sys.exit(0)
        sentences = adjust_sentences(sentences, output_file_path)
        with open(output_file_path, 'a') as output_file:
            for i, sentence in enumerate(sentences):
                print(f'Translating sentence {i+1}/{len(sentences)}')
                translated_text = safe_translate(sentence, model_name)
                output_file.write(translated_text + "\n")

print(f'Time taken: {time() - s} seconds')