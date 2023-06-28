#!/usr/bin/env python3
from transformers import T5Tokenizer #, T5ForConditionalGeneration

from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

#HFMODEL = "google/flan-t5-xxl"; #9.45G
HFMODEL = "google/flan-t5-base"; #990MB

def query_tokens(text):
    tokenizer = T5Tokenizer.from_pretrained(HFMODEL)
    #model = T5ForConditionalGeneration.from_pretrained(HFMODEL, device_map="auto")
    return tokenizer.tokenize(text), tokenizer(text).input_ids #.to("cuda")

text = "As she said this, she looked down at her hands, and was surprised to find that she had put on one of the rabbit's little gloves while she was talking."

response = query_tokens(text)

print("TEXT:")
print(text)
print("----")
print("TOKENS:")
print(response[0])
print("TOKENIDs:")
print(response[1])