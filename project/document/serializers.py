from rest_framework import serializers
from .models import Document

class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = ['id', 'title', 'content', 'uploaded_at']

    def create(self, validated_data):
        # Aquí puedes procesar el archivo si es necesario
        return super().create(validated_data)
