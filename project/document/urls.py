from django.urls import path
from .views import DocumentUploadView, DocumentSearchView, DocumentSummarizeView

urlpatterns = [
    path('upload/', DocumentUploadView.as_view(), name='upload_document'),
    path('search/', DocumentSearchView.as_view(), name='search_document'),
    path('<int:pk>/summarize/', DocumentSummarizeView.as_view(), name='summarize_document'),
]
