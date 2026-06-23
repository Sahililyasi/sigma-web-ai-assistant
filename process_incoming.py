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
    top_results=5
    max_indices=similarities.argsort()[::-1][0:top_results]

    #print(max_indices)
    new_df=df.loc[max_indices]
    #print(new_df[["title","number","text"]])
    new_df = new_df.copy()
    new_df["similarity"] = similarities[max_indices]


    # print("\n===== RETRIEVED CHUNKS =====\n")
    # # for index, row in new_df.iterrows(): TERMINAL MEIN DEKHNE KE LIYE CHUNKS AND ALL
    # #     print(f"\nVIDEO: {row['title']}")
    # #     print(f"TIME: {row['start']} - {row['end']}")
    # #     print(row['text'])
    # #     print("-" * 80)

    prompt = f"""
You are a retrieval-based course assistant for the Sigma Web Development Course.

Your knowledge is LIMITED ONLY to the retrieved chunks provided below.

Retrieved Chunks:
{new_df[["title","number","start","end","text"]].to_json(orient="records")}

---

User Question:
{incoming_query}

Instructions:

1. Use ONLY information present in the retrieved chunks.
2. Never use outside knowledge.
3. Never invent video numbers, video titles, timestamps, topics, explanations, examples, or recommendations.
4. If the retrieved chunks explain the requested topic, answer using those explanations.
5. If multiple retrieved chunks contain useful information, combine the relevant information into a single answer.
6. Prefer chunks that DEFINE or EXPLAIN a concept over chunks that only mention it.
7. Do not rely on a chunk only because it has the highest similarity score.
8. Evaluate all retrieved chunks before answering.
9. If the topic is only mentioned but not explained in the retrieved chunks, return exactly:

The topic is only mentioned in the retrieved content.

10. If the answer cannot be supported by the retrieved chunks, return exactly:

I could not find this information in the course.

11. Never output:
- FOUND
- NOT_FOUND
- YES
- NO

12. Do not mention chunk numbers, chunk rankings, similarity scores, or retrieval details in your answer.

13. Answer naturally, clearly, and concisely.

Answer:
"""

    with open("prompt.txt","w")as f:
        f.write(prompt)

    response=inference(prompt)["response"]
    
    #print(response)

    with open("response.txt","w")as f:
        f.write(response)
    # for index,item in new_df.iterrows():
    #     print (index, item["title"],item["number"],item["text"],item["start"],item["end"])


    return {
    "answer": response,
    "sources": new_df[
        ["number", "title", "start", "end", "similarity", "text"]
    ].to_dict("records")
}
if __name__ == "__main__":
    question = input("Ask a Question: ")
    answer = ask_question(question)