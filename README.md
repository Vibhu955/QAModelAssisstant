🧠 CounselChat Q&A Assistant

This project is an intelligent mental health Q&A assistant that retrieves, reranks, and summarizes therapist answers from a curated dataset using state-of-the-art NLP models.

## 🚀 Features

- ✅ Embedding using `all-mpnet-base-v2` (Sentence Transformers)
- ✅ Efficient similarity search via FAISS indexing
- ✅ Reranking with `ms-marco-MiniLM-L-6-v2` CrossEncoder
- ✅ Answer summarization using `distilbart-cnn-12-6`
- ✅ Fast response interface powered by Gradio
- ✅ Therapist profile link generation (PsychologyToday)

---

## 🛠️ How It Works

1. **Embedding**  
   User query and questions from the dataset are embedded using `all-mpnet-base-v2`.

2. **Similarity Search (FAISS)**  
   FAISS performs fast approximate nearest neighbor search to retrieve top similar questions.

3. **Reranking**  
   CrossEncoder reranks retrieved results based on their relevance to the query.

4. **Summarization**  
   The top answers are summarized using a BART-based model for concise results.

5. **UI with Gradio**  
   Users can input their questions and view top 3 therapist responses with summaries and links.

---