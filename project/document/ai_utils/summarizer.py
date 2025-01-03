from transformers import pipeline

# Configurar el modelo de resumen
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text: str, max_length: int = 130, min_length: int = 30) -> str:
    """Genera un resumen del texto dado."""
    try:
        return summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
    except Exception as e:
        raise ValueError(f"Error generando resumen: {str(e)}")
