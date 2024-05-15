from transformers import MarianMTModel, MarianTokenizer
import os


def translate(text, model_name):
    # Load the tokenizer and model
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Tokenize the text
    batch = tokenizer([text], return_tensors="pt", padding=True)

    # Generate translation outputs
    translated = model.generate(**batch)
    translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    
    return translated_text

def split_text(text):
    return text.split('\n')

# Example usage
input_text = "The report also said the monitoring team had received information that two senior Islamic State commanders, Abu Qutaibah and Abu Hajar al-Iraqi, had recently arrived in Afghanistan from the Middle East."

model_name = "Helsinki-NLP/opus-mt-en-de"
translated_text = translate(input_text, model_name)
print(f'Translated text: {translated_text}')