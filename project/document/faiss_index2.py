import faiss
import os
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Configuración
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "faiss_index.bin"
dimension = 384  # Dimensión del vector de embeddings

# Inicializa el modelo
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Índice de FAISS
index = None


def get_embedding(text):
    """Genera un embedding para el texto dado."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()


def add_to_index(embedding, doc_id):
    """Agrega un embedding al índice."""
    global index
    index.add_with_ids(np.array([embedding]), np.array([doc_id]))


def save_index():
    """Guarda el índice en un archivo."""
    global index
    faiss.write_index(index, INDEX_PATH)


def load_index():
    """Carga el índice desde un archivo, o crea uno nuevo si no existe."""
    global index
    if os.path.exists(INDEX_PATH):
        print("Cargando índice FAISS existente...")
        index = faiss.read_index(INDEX_PATH)
    else:
        print("No se encontró el archivo FAISS, creando un índice nuevo...")
        index = faiss.IndexFlatL2(dimension)  # Índice vacío
