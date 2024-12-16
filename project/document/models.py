from django.db import models

class Document(models.Model):
    title = models.CharField(max_length=255)
    content = models.TextField()
    embedding = models.BinaryField()  # Guardaremos el vector de embeddings
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title
