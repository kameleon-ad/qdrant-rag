from __future__ import annotations

import json
from pathlib import Path

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams

from config import Config


class NCLAgent:
    def __init__(self, data_path: Path | str):
        self.qdrant_client = QdrantClient(
            url=Config.QDRANT_CLUSTER, api_key=Config.QDRANT_API_KEY
        )
        self.openai_client = OpenAI()
        self.embedding_model = Config.EMBEDDING_MODEL
        self.collections = ["introduction", "heading", "include", "exclude"]

        data = json.load(open(data_path, "r", encoding="utf8"))

        self.data = {class_item["class_id"]: class_item for class_item in data}

    def __call__(self, product_description: str, limit_candidate: int = 2):
        return self.infer(product_description, limit_candidate)

    def search_per_collection(
        self, query_vector: list[float], collection: str, limit_candidate: int = 2
    ):
        return self.qdrant_client.search(
            collection_name=collection,
            search_params=SearchParams(hnsw_ef=128, exact=False),
            query_vector=query_vector,
            limit=limit_candidate,
        )

    def get_candidate_from_text(self, text: str, limit_candidate: int = 2):
        candidate = set()
        for collection in self.collections:
            candidate = candidate | set(
                (
                    scored_point.payload["class_id"]
                    for scored_point in self.search_per_collection(
                        query_vector=self.text_to_embedding(text),
                        collection=collection,
                        limit_candidate=limit_candidate,
                    )
                )
            )
        return candidate

    def text_to_embedding(self, text: str):
        return (
            self.openai_client.embeddings.create(
                input=[text], model=self.embedding_model
            )
            .data[0]
            .embedding
        )

    def build_template_per_class(self, class_id: int):
        class_data = self.data[class_id]
        messages = []

        messages.append(class_data["introduction"])
        t = f"The class {class_id} includes "
        for head in class_data["heading"]:
            t = t + head + ","
        messages.append(t)
        t = f"The class {class_id} includes particulary "
        for include in class_data["include"]:
            t = t + include + ","
        messages.append(t)
        t = f"The class {class_id} excludes particulary "
        for exclude in class_data["exclude"]:
            t = t + exclude + ","
        messages.append(t)
        return messages

    def infer(self, description: str, limit_candidate: int = 2):
        messages = []

        for candidate_id in self.get_candidate_from_text(description, limit_candidate):
            messages.extend(
                [
                    {"role": "user", "content": message}
                    for message in self.build_template_per_class(candidate_id)
                ]
            )

        messages.extend(
            [
                {
                    "role": "user",
                    "content": "I'm working on NICE classification for the intellectual property",
                },
                {"role": "user", "content": "Here is the description of the product"},
                {"role": "user", "content": description},
                {"role": "user", "content": "What is the class of this product?"},
            ]
        )

        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )

        return response.choices[0].message.content


NCL_AGENT = NCLAgent("data/output.json")
