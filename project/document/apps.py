from django.apps import AppConfig
from .faiss_index import load_index




class DocumentConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'document'

    def ready(self):
        load_index()