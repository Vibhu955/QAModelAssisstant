import pandas as pd
import torch
import os
import re
import faiss
import hashlib
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
import gradio as gr
from concurrent.futures import ThreadPoolExecutor

# === Load and clean dataset ===
df = pd.read_csv("20220401_counsel_chat.zip")
df.dropna(subset=["questionText", "questionTitle", "answerText", "therapistInfo", "therapistURL"], inplace=True)
df.reset_index(drop=True, inplace=True)
df["combinedQuestion"] = df["questionTitle"].str.strip() + " - " + df["questionText"].str.strip()
print(df.head())

# === Load models ===
device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=0 if torch.cuda.is_available() else -1)


# === Embedding + FAISS ===
EMBED_PATH = "mpnet_embeddings.pt"
INDEX_PATH = "faiss_index.index"
summary_cache = {}
query_cache = {}

if not os.path.exists(EMBED_PATH):
    embeddings = embed_model.encode(df["combinedQuestion"].tolist(), convert_to_tensor=True, show_progress_bar=True, normalize_embeddings=True)
    torch.save(embeddings, EMBED_PATH)
else:
    embeddings = torch.load(EMBED_PATH)

dimension = embeddings.shape[1]
if not os.path.exists(INDEX_PATH):
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.cpu().numpy())
    faiss.write_index(index, INDEX_PATH)
else:
    index = faiss.read_index(INDEX_PATH)


# === Helpers ===
def hash_text(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

def build_psychologytoday_url(name: str) -> str:
    cleaned = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', name)
    cleaned = re.sub(r'[\d/]+', '', cleaned)
    capitalized_words = re.findall(r'\b[A-Z][a-zA-Z\-]*\b', cleaned)
    first_two = capitalized_words[:2] if capitalized_words else ["Therapist"]
    query = '+'.join(first_two)
    return f"https://www.psychologytoday.com/us/therapists?search={query}"


#display ans
def format_answer(idx, row, summary):
    therapist = row.therapistInfo.strip()
    topic = row.topic.strip() if pd.notna(row.topic) else "Unknown"
    therapist_url = build_psychologytoday_url(therapist)
    views = int(row.views) if pd.notna(row.views) else 0
    upvotes = int(row.upvotes) if pd.notna(row.upvotes) else 0
    answer = row.answerText.strip()

    return f"""
💡 *Topic*: `{topic}`

### 🔷 Answer {idx + 1}
👩‍⚕️ **Therapist**: {therapist}  
🔗 [PsychologyToday Profile]({therapist_url})  
⚠️ _We link to public therapist listings for convenience. We do not verify or endorse them._

#### 📝 Summary:
{summary}

<details>
<summary>📖 Click to view full answer</summary>

{answer}

</details>

👁️ **Views**: {views}
👍 **Upvotes**: {upvotes}
"""

def summarize(text):
    key = hash_text(text)
    if key in summary_cache:
        return summary_cache[key]
    short_text = text[:512] if len(text) > 512 else text
    result = summarizer(short_text)[0]["summary_text"]
    summary_cache[key] = result
    return result


# === Main pipeline ===
def process_query(query: str) -> str:
    if not query:
        return "⚠️ Please enter a query..!!"

    query_key = hash_text(query)
    if query_key in query_cache:
        return query_cache[query_key]

    query_embedding = embed_model.encode([query], convert_to_tensor=True, normalize_embeddings=True)
    D, I = index.search(query_embedding.cpu().numpy(), k=50)
    filtered_indices = [i for i, score in zip(I[0], D[0]) if score >= 0.4]

    if not filtered_indices:
        return "❌ No relevant answers found."

    top_df = df.iloc[filtered_indices].copy()
    pairs = [[query, f"{row.combinedQuestion} {row.answerText}"] for row in top_df.itertuples()]
    rerank_scores = reranker.predict(pairs)
    top_df["rerank_score"] = rerank_scores
    top_df = top_df.sort_values(by=["rerank_score", "views", "upvotes"], ascending=[False, False, False]).head(3)

    summaries, results = [], []

    with ThreadPoolExecutor() as executor:
        sum_futures = [executor.submit(summarize, row.answerText) for row in top_df.itertuples()]
        for idx, (row, future) in enumerate(zip(top_df.itertuples(), sum_futures)):
            sum_text = future.result()
            summaries.append(sum_text)
            results.append(format_answer(idx, row, sum_text))

    final_summary = summarizer(" ".join(summaries))[0]["summary_text"]
    full_output = "\n\n---\n\n".join(results) + f"\n\n---\n\n🧠 **Final Summary**:\n{final_summary}"
    query_cache[query_key] = full_output
    return full_output


import pickle

# Save SentenceTransformer
with open("embed_model.pkl", "wb") as f:
    pickle.dump(embed_model, f)

# Save CrossEncoder
with open("reranker.pkl", "wb") as f:
    pickle.dump(reranker, f)

# Save HuggingFace summarizer pipeline
with open("summarizer.pkl", "wb") as f:
    pickle.dump(summarizer, f)

# Save FAISS index and DataFrame
torch.save(embeddings, "embeddings.pt")
df.to_pickle("df.pkl")
faiss.write_index(index, "faiss_index.index")



# === Launch Gradio App ===
# gr.Interface(
#     fn=process_query,
#     inputs=gr.Textbox(label="❓ Ask your mental health question here", placeholder="e.g. How can I deal with anxiety about work?", lines=2),
#     outputs=gr.Markdown(label="🩺 Top Therapist Answers"),
#     title="🧘 CounselChat Q&A Assistant",
#     description="Explain in brief about your problems.",
#     allow_flagging="never"
# ).launch()



