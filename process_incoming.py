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

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })

    response = r.json()
    return response

df=joblib.load('embeddings.joblib')
def ask_question(incoming_query):
    question_embedding = create_embedding([incoming_query])[0]



    # Find similarity of question_embedding with other_embeddings

    similarities = cosine_similarity(np.vstack(df['embedding']),[question_embedding]).flatten()

    #print(similarities)
    top_results=3
    max_indices=similarities.argsort()[::-1][0:top_results]

    #print(max_indices)
    new_df=df.loc[max_indices]
    #print(new_df[["title","number","text"]])

    prompt = f'''You are a retrieval-based course assistant for the Sigma Web Development Course.

Your knowledge is LIMITED ONLY to the retrieved chunks provided below.

Retrieved Chunks:
{new_df[["title","number","start","end","text"]].to_json(orient="records")}

---

User Question:
{incoming_query}

Instructions:

1. Answer ONLY using information present in the retrieved chunks.
2. Never use outside knowledge.
3. Never invent video numbers, video titles, timestamps, topics, explanations, or recommendations.
4. If the retrieved chunks explain or discuss the requested topic, answer using only those chunks.
5. If the retrieved chunks only mention the topic but do not explain it, clearly say that the topic is only mentioned in the retrieved content.
6. If the answer cannot be supported by the retrieved chunks, return exactly:

I could not find this information in the course.

7. Never output reasoning labels such as:
   FOUND
   NOT_FOUND
   YES
   NO

8. Do not provide guesses, assumptions, or alternative recommendations.

Answer naturally and concisely.

    '''

    with open("prompt.txt","w")as f:
        f.write(prompt)

    response=inference(prompt)["response"]
    
    #print(response)

    with open("response.txt","w")as f:
        f.write(response)
    # for index,item in new_df.iterrows():
    #     print (index, item["title"],item["number"],item["text"],item["start"],item["end"])
    return response



if __name__ == "__main__":
    question = input("Ask a Question: ")
    answer = ask_question(question)