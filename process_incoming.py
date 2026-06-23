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
    new_df = new_df.copy()
    new_df["similarity"] = similarities[max_indices]


    # print("\n===== RETRIEVED CHUNKS =====\n")
    # # for index, row in new_df.iterrows(): TERMINAL MEIN DEKHNE KE LIYE CHUNKS AND ALL
    # #     print(f"\nVIDEO: {row['title']}")
    # #     print(f"TIME: {row['start']} - {row['end']}")
    # #     print(row['text'])
    # #     print("-" * 80)

    prompt = f'''You are a retrieval-based course assistant for the Sigma Web Development Course.

Your knowledge is LIMITED ONLY to the retrieved chunks provided below.

Retrieved Chunks:
{new_df[["title","number","start","end","text"]].to_json(orient="records")}

---

User Question:
{incoming_query}

Instructions:

You MUST follow these rules exactly.

1. Use ONLY information explicitly present in the retrieved chunks.
2. Treat the retrieved chunks as the complete course knowledge.
3. Do NOT use any outside knowledge, even if you know the answer.
4. Do NOT explain, expand, infer, summarize, or add details that are not explicitly stated in the retrieved chunks.
5. Do NOT use general knowledge examples.
6. Do NOT mention concepts, tools, websites, technologies, or examples unless they appear in the retrieved chunks.
7. If the retrieved chunks explain the topic, answer only from those chunks.
8. If the retrieved chunks only mention the topic but do not explain it, say:
   "The topic is only mentioned in the retrieved content."
9. If the answer is not supported by the retrieved chunks, return exactly:
   "I could not find this information in the course."
10. Never invent:
    - Video numbers
    - Video titles
    - Timestamps
    - Definitions
    - Explanations
    - Examples
    - Recommendations
11. Never output FOUND, NOT_FOUND, YES, or NO.
12. Your answer must be fully supported by the retrieved chunks.

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

    
    return {
    "answer": response,
    "sources": new_df[
        ["number", "title", "start", "end", "similarity", "text"]
    ].to_dict("records")
}
if __name__ == "__main__":
    question = input("Ask a Question: ")
    answer = ask_question(question)