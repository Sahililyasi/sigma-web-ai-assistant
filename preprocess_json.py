import requests
import os
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib


def create_embedding(text_list, batch_size=50):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    all_embeddings = []

    for start in range(0, len(text_list), batch_size):
        batch = text_list[start:start + batch_size]
        r = requests.post("http://localhost:11434/api/embed", json={
            "model": "bge-m3",
            "input": batch
        })

        try:
            response = r.json()
        except requests.JSONDecodeError as exc:
            raise RuntimeError(f"Ollama returned invalid JSON: {r.text}") from exc

        if "embeddings" not in response:
            raise RuntimeError(
                f"Ollama did not return embeddings for batch {start}-{start + len(batch)}. "
                f"Status: {r.status_code}. Response: {response}"
            )

        all_embeddings.extend(response["embeddings"])

    return all_embeddings


jsons = os.listdir("jsons")  # List all the jsons 
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    print(f"Creating Embeddings for {json_file}")
    embeddings = create_embedding([c['text'] for c in content['chunks']])
       
    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk) 
    

df = pd.DataFrame.from_records(my_dicts)
# save this dataframe
joblib.dump(df,'embeddings.joblib')


