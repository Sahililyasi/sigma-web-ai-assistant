import whisper
import json
import os

model = whisper.load_model("large-v2")

rag_all_audios = os.listdir("rag_all_audios")

for audio in rag_all_audios:

    if "_" not in audio:
        print(f"Skipping: {audio}")
        continue

    number = audio.split("_")[0]
    title = audio.split("_")[1][:-4]

    print(number, title)

    result = model.transcribe(
        audio=f"rag_all_audios/{audio}",
        language="hi",
        task="translate",
        word_timestamps=False
    )

    chunks = []

    for segment in result["segments"]:
        chunks.append({
            "number": number,
            "title": title,
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"]
        })

    chunks_with_metadata = {
        "chunks": chunks,
        "text": result["text"]
    }

    with open(f"jsons/{audio}.json", "w") as f:
        json.dump(chunks_with_metadata, f, indent=4)

print("Done!")