from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from .models import Document
from .serializers import DocumentSerializer
from .ai_utils import summarize_text, search_documents
from .faiss_index import add_to_index, save_index, get_embedding


class DocumentUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        print("Datos recibidos:", request.data)
        print("Archivos recibidos:", request.FILES)

        file = request.FILES.get("file")
        content = ""

        # Procesar el archivo si existe
        if file:
            if file.name.endswith(".pdf"):
                # Procesar PDF
                try:
                    reader = PdfReader(file)
                    content = " ".join(page.extract_text() for page in reader.pages)
                except Exception as e:
                    return Response({"error": f"Error procesando PDF: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
            elif file.name.endswith(".docx"):
                # Procesar Word
                try:
                    doc = DocxDocument(file)
                    content = " ".join(paragraph.text for paragraph in doc.paragraphs)
                except Exception as e:
                    return Response({"error": f"Error procesando Word: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
            else:
                # Archivo no soportado
                return Response({"error": "Formato de archivo no soportado. Solo se aceptan PDF y Word."}, status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response({"error": "No se envió ningún archivo."}, status=status.HTTP_400_BAD_REQUEST)

        # Crear datos para el serializador
        data = {
            "title": request.data.get("title"),
            "content": content,  # El contenido extraído del archivo
        }
        serializer = DocumentSerializer(data=data)

        if serializer.is_valid():
            doc = serializer.save()

            # Generar embeddings y agregar al índice
            if doc.content:
                embedding = get_embedding(doc.content)
                add_to_index(embedding, doc.id)
                save_index()

            return Response(serializer.data, status=status.HTTP_201_CREATED)

        # Mostrar errores del serializador en la consola
        print("Errores del serializador:", serializer.errors)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class DocumentSearchView(APIView):
    def get(self, request):
        query = request.GET.get("query", "")
        if not query:
            return Response({"error": "Query parameter is required"}, status=status.HTTP_400_BAD_REQUEST)
        
        indices, distances = search_documents(query)
        documents = Document.objects.filter(id__in=indices.flatten())
        serializer = DocumentSerializer(documents, many=True)
        return Response({"results": serializer.data, "distances": distances.tolist()})


class DocumentSummarizeView(APIView):
    def get(self, request, pk):
        try:
            document = Document.objects.get(pk=pk)
        except Document.DoesNotExist:
            return Response({"error": "Document not found"}, status=status.HTTP_404_NOT_FOUND)
        
        summary = summarize_text(document.content)
        return Response({"title": document.title, "summary": summary})
