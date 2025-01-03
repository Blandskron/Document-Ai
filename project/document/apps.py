from django.apps import AppConfig
from .ai_utils.faiss_index import initialize_faiss_index


class DocumentConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'document'

    def ready(self):
        """
        Este método se ejecuta cuando la configuración de la app está lista.
        Aquí inicializamos el índice FAISS.
        """
        print("Inicializando índice FAISS...")
        initialize_faiss_index()
