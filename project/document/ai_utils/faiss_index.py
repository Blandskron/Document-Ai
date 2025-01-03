import faiss
import os
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Configuración global
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "faiss_index.bin"
EMBEDDING_DIMENSION = 384

# Inicialización de modelos
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Índice FAISS
index = None

def initialize_faiss_index():
    """Inicializa el índice FAISS."""
    global index
    if os.path.exists(INDEX_PATH):
        print("Cargando índice FAISS existente...")
        index = faiss.read_index(INDEX_PATH)
    else:
        print("Creando un nuevo índice FAISS...")
        # Usar IndexIDMap para permitir IDs personalizados
        index = faiss.IndexIDMap(faiss.IndexFlatL2(EMBEDDING_DIMENSION))

def get_embedding(text: str) -> np.ndarray:
    """Genera un embedding para el texto dado."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    # Generar vector promedio
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()


def add_to_index(embedding: np.ndarray, doc_id: int):
    """Agrega un embedding al índice con su ID asociado."""
    global index
    if index is None:
        initialize_faiss_index()
    
    # Asegurarse de que el embedding sea bidimensional
    if len(embedding.shape) == 1:
        embedding = embedding.reshape(1, -1)
    
    # Agregar el embedding con su ID
    index.add_with_ids(embedding, np.array([doc_id], dtype=np.int64))


def save_index():
    """Guarda el índice FAISS en un archivo."""
    if index:
        faiss.write_index(index, INDEX_PATH)

def search_documents(query: str, top_k: int = 5):
    """Busca documentos en el índice FAISS usando una consulta."""
    global index
    if index is None:
        raise ValueError("El índice FAISS no está inicializado.")
    
    # Generar el vector de consulta
    query_vector = get_embedding(query)
    
    # Asegurarse de que el vector de consulta tenga forma (1, d)
    if len(query_vector.shape) == 1:
        query_vector = query_vector.reshape(1, -1)
    
    # Realizar la búsqueda
    try:
        distances, indices = index.search(query_vector, top_k)
        return indices.flatten().tolist(), distances.flatten().tolist()
    except Exception as e:
        raise RuntimeError(f"Error al buscar en el índice FAISS: {str(e)}")


