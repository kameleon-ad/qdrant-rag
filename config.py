import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
    QDRANT_CLUSTER = os.environ["QDRANT_CLUSTER"]
    EMBEDDING_MODEL = "text-embedding-3-large"
    EMBEDDING_DIMENSION = 3072
