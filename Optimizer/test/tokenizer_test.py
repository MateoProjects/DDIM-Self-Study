from transformers import BertTokenizer, CLIPTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer_clip = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

def encode_text(prompt):
    text = tokenizer.encode(prompt)
    text_2 = tokenizer_clip.tokenize(prompt)
    text_2 = tokenizer_clip.convert_tokens_to_ids(text_2)
    print(text, text_2)
    return text,text_2

def decode_tokens(tokens, tokens2):
    print(tokenizer.decode(tokens))
    tokens2 = tokenizer_clip.convert_ids_to_tokens(tokens2)
    print(tokenizer_clip.convert_tokens_to_string(tokens2))

text, text2 = encode_text("This is a test")
decode_tokens(text, text2)