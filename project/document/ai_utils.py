from transformers import pipeline

# Configurar resumen
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    return summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']

# Búsqueda en FAISS
from .faiss_index import get_embedding, index

def search_documents(query, top_k=5):
    query_vector = get_embedding(query)
    distances, indices = index.search(np.array([query_vector]), top_k)
    return indices, distances
