from transformers import BertTokenizer, CLIPTokenizer


tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

def text_to_id(prompt):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompt))

def id_to_text(ids):
    return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(ids))

def encoding_text(ids):
    pass

def decoding_text(encoders):
    pass



