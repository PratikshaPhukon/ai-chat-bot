from vertexai import init
from vertexai.preview.language_models import TextEmbeddingModel
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from vertexai.preview.generative_models import GenerativeModel
import os


def get_credentials():
    service_account_path = os.getenv("SERVICE_ACCOUNT_PATH")
    target_audience = os.getenv("TARGET_AUDIENCE")
    credentials = service_account.Credentials.from_service_account_file(
        service_account_path
    )
    token_credentials = service_account.IDTokenCredentials.from_service_account_file(
        service_account_path, target_audience=target_audience
    )
    token_credentials.refresh(Request())
    return credentials, token_credentials


# Initialize Vertex AI
def initialize_vertex_ai(credentials):
    project_id = os.getenv("PROJECT_ID")
    location = os.getenv("LOCATION")
    init(project=project_id, location=location, credentials=credentials)


# Get embedding vector for a question
def get_query_vector(question: str) -> list:
    embedding_model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")
    # gemini-embedding-001 returns 3072 dimensions with MRL (Matryoshka Representation Learning)
    # This provides higher quality embeddings for better retrieval performance
    return embedding_model.get_embeddings([question])[0].values


def generate_answer_stream(context: str, question: str):
    model = GenerativeModel("gemini-2.5-pro")
    prompt = f"""
    You are a helpful AI assistant. Use the context below to answer the question accurately using **Markdown formatting**. 
    If the answer benefits from bullet points, tables, or code blocks, use them.
    Avoid any introduction, summary, or repetition of the question—just provide a clean, formatted answer.

    ### Context:
    {context}

    ### Question:
    {question}

    ### Markdown Answer:
    """
    chat = model.start_chat(history=[])
    response = chat.send_message(prompt, stream=True)
    for chunk in response:
        if chunk.text:
            yield chunk.text