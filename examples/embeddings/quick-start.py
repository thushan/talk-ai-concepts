import os
import pandas as pd
import requests
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HFMODEL = "sentence-transformers/all-MiniLM-L6-v2";
HFA_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HFMODEL}";
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

def query_embeddings(texts):
    req_body = {"inputs": texts, "options": {"wait_for_model": True}}
    response = requests.post(HFA_URL, headers=HEADERS, json=req_body)
    return response.json()

texts = ["Who is the prime minister of Australia?",
         "What is 1 + 1 and why is it Window?",
         "Can I get my Subaru serviced at an Audi Dealership?",
         "Are people from Sydney a bit posh?"
        ]

response = query_embeddings(texts)
embeddings = pd.DataFrame(response)

with pd.option_context('display.max_rows', None, 
                       'display.max_columns', None):
    print(embeddings)

embeddings.to_csv("output.csv", index=False)