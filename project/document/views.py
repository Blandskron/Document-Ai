from rest_framework.parsers import MultiPartParser, FormParser
from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiTypes, OpenApiResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from .models import Document
from .serializers import DocumentSerializer
from .ai_utils.summarizer import summarize_text
from .ai_utils.faiss_index import add_to_index, save_index, get_embedding, search_documents

class DocumentUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    @extend_schema(
        request={
            'multipart/form-data': {
                'type': 'object',
                'properties': {
                    'title': {'type': 'string', 'example': 'Mi Documento'},
                    'file': {'type': 'string', 'format': 'binary'},
                },
            },
        },
        responses={
            201: {'description': 'Documento subido correctamente.'},
            400: {'description': 'Errores en la solicitud.'},
        },
    )
    def post(self, request):
        file = request.FILES.get("file")
        title = request.data.get("title")
        if not file or not title:
            return Response({"error": "Título y archivo son obligatorios."}, status=status.HTTP_400_BAD_REQUEST)

        content = self._process_file(file)
        if not content:
            return Response({"error": "Error al procesar el archivo."}, status=status.HTTP_400_BAD_REQUEST)

        serializer = DocumentSerializer(data={"title": title, "content": content})
        if serializer.is_valid():
            doc = serializer.save()
            try:
                embedding = get_embedding(doc.content)
                add_to_index(embedding, doc.id)
                save_index()
            except Exception as e:
                # Eliminar el documento si falla el índice
                doc.delete()
                return Response({"error": f"Error al agregar al índice: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @staticmethod
    def _process_file(file) -> str:
        """Procesa archivos PDF o Word y extrae su contenido."""
        try:
            if file.name.endswith(".pdf"):
                reader = PdfReader(file)
                return " ".join(page.extract_text() for page in reader.pages)
            elif file.name.endswith(".docx"):
                doc = DocxDocument(file)
                return " ".join(paragraph.text for paragraph in doc.paragraphs)
        except Exception as e:
            print(f"Error procesando archivo: {e}")
        return ""


@extend_schema(
    parameters=[
        OpenApiParameter(
            name="query", type=OpenApiTypes.STR, location=OpenApiParameter.QUERY,
            description="Texto de búsqueda para encontrar documentos."
        )
    ],
    responses={
        200: OpenApiResponse(
            response=OpenApiTypes.OBJECT,
            description="Resultados de búsqueda con documentos y distancias."
        ),
        400: OpenApiResponse(
            response=OpenApiTypes.OBJECT,
            description="Error debido a parámetros incorrectos."
        ),
        500: OpenApiResponse(
            response=OpenApiTypes.OBJECT,
            description="Error interno del servidor durante la búsqueda."
        ),
    }
)
class DocumentSearchView(APIView):
    def get(self, request):
        query = request.GET.get("query", "")
        if not query:
            return Response({"error": "Query parameter is required"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            indices, distances = search_documents(query)
            if not indices:  # Validar que existan resultados
                return Response({"results": [], "distances": []}, status=status.HTTP_200_OK)
            
            documents = Document.objects.filter(id__in=indices)
            serializer = DocumentSerializer(documents, many=True)
            return Response({"results": serializer.data, "distances": distances})
        except ValueError as e:
            print(f"ValueError: {e}")
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            print(f"Error inesperado: {e}")
            return Response({"error": f"Error durante la búsqueda: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



@extend_schema(
    responses={200: 'Resumen del documento.'}
)
class DocumentSummarizeView(APIView):
    def get(self, request, pk):
        try:
            document = Document.objects.get(pk=pk)
            summary = summarize_text(document.content)
            return Response({"title": document.title, "summary": summary})
        except Document.DoesNotExist:
            return Response({"error": "Document not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": f"Error generating summary: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
