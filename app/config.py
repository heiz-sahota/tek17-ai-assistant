"""Settings and Azure clients, shared by the API, ingestion and evaluation.

Uses Microsoft Entra ID through DefaultAzureCredential: your `az login` locally,
the managed identity on Azure. Falls back to API keys only if they are set in .env.
"""
import os
from functools import lru_cache

from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
CHAT_DEPLOYMENT = os.environ.get("CHAT_DEPLOYMENT", "chat")
EMBEDDING_DEPLOYMENT = os.environ.get("EMBEDDING_DEPLOYMENT", "embed")
EMBEDDING_DIMENSIONS = int(os.environ.get("EMBEDDING_DIMENSIONS", "1536"))
SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX", "tek17")


@lru_cache
def credential() -> DefaultAzureCredential:
    return DefaultAzureCredential()


@lru_cache
def openai_client() -> AzureOpenAI:
    key = os.environ.get("AZURE_OPENAI_API_KEY")
    if key:
        return AzureOpenAI(azure_endpoint=OPENAI_ENDPOINT, api_version=OPENAI_API_VERSION, api_key=key)
    token = get_bearer_token_provider(credential(), "https://cognitiveservices.azure.com/.default")
    return AzureOpenAI(azure_endpoint=OPENAI_ENDPOINT, api_version=OPENAI_API_VERSION, azure_ad_token_provider=token)


def _search_credential():
    key = os.environ.get("AZURE_SEARCH_API_KEY")
    return AzureKeyCredential(key) if key else credential()


@lru_cache
def search_client() -> SearchClient:
    return SearchClient(SEARCH_ENDPOINT, SEARCH_INDEX, _search_credential())


@lru_cache
def index_client() -> SearchIndexClient:
    return SearchIndexClient(SEARCH_ENDPOINT, _search_credential())


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with the Foundry embedding deployment."""
    result = openai_client().embeddings.create(model=EMBEDDING_DEPLOYMENT, input=texts)
    return [item.embedding for item in result.data]
