from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import TypedDict, DefaultDict

from colorama import Fore, init
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.models import Distance, VectorParams
from tqdm import tqdm

load_dotenv()


class NiceClass(TypedDict):
    class_id: str
    heading: list[str]
    introduction: str
    include: list[str]
    exclude: list[str]
    good_or_service: list[str]


class PayloadClass(TypedDict):
    class_id: int


def defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        # Convert the defaultdict itself
        d = {key: defaultdict_to_dict(value) for key, value in d.items()}
    return dict(d)


QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
QDRANT_CLUSTER = os.environ["QDRANT_CLUSTER"]
COLLECTION_INFO = {
    "heading": {
        "collection_name": "heading",
    },
    "introduction": {
        "collection_name": "introduction",
    },
    "include": {
        "collection_name": "include",
    },
    "exclude": {
        "collection_name": "exclude",
    },
    "good_or_service": {
        "collection_name": "good_or_service",
    },
}
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSION = 3072
OPENAI_LOGGER = "OPENAI_LOGGER"
QDRANT_LOGGER = "QDRANT_LOGGER"

QDRANT_CLIENT = QdrantClient(url=QDRANT_CLUSTER, api_key=QDRANT_API_KEY)
OPENAI_CLIENT = OpenAI()


def setup_logger():
    # Configure logging format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Setup logger for OpenAI
    openai_logger = logging.getLogger(OPENAI_LOGGER)
    openai_file_handler = logging.FileHandler("openai.log")
    openai_file_handler.setFormatter(formatter)
    openai_logger.setLevel(
        logging.DEBUG
    )  # Log at any level (DEBUG is the lowest operational level)
    openai_logger.addHandler(openai_file_handler)

    # Setup logger for Qdrant
    qdrant_logger = logging.getLogger(QDRANT_LOGGER)
    qdrant_file_handler = logging.FileHandler("qdrant_logger.log")
    qdrant_file_handler.setFormatter(formatter)
    qdrant_logger.setLevel(logging.DEBUG)  # Same here, log everything
    qdrant_logger.addHandler(qdrant_file_handler)


def log(logger_name, msg, level=logging.DEBUG, *args, **kwargs):
    logging.getLogger(logger_name).log(level, msg, *args, **kwargs)


def get_vector(text: str) -> list:
    log(OPENAI_LOGGER, f"Getting vector for text: {text}")
    result = (
        OPENAI_CLIENT.embeddings.create(input=[text], model=EMBEDDING_MODEL)
        .data[0]
        .embedding
    )
    log(OPENAI_LOGGER, f"Got vector for text: {text}")
    return result


def process_per_class(
    nice_class: NiceClass, qdrant_store: DefaultDict[str, DefaultDict[str, tuple]]
):
    payload = PayloadClass({"class_id": nice_class["class_id"]})

    introduction = nice_class["introduction"].lower()
    qdrant_store["introduction"][introduction] = (
        get_vector(introduction),
        payload,
    )
    for field in tqdm(
        ["heading", "include", "exclude", "good_or_service"],
        desc=Fore.GREEN,
        position=1,
        leave=False,
    ):
        for heading_item in tqdm(
            nice_class[field], desc=Fore.BLUE, position=2, leave=False
        ):
            heading_item = heading_item.lower()
            vector = get_vector(heading_item)
            qdrant_store[field][heading_item] = (vector, payload)


def push_qdrant_store(
    qdrant_store: DefaultDict[str, DefaultDict[str, tuple]], skip_fields: list
):
    for field in qdrant_store:
        if field in skip_fields:
            continue
        index = 0
        points = []
        collection_name = COLLECTION_INFO[field]["collection_name"]
        for item in qdrant_store[field]:
            vector, payload = qdrant_store[field][item]
            points.append(
                PointStruct(
                    id=index,
                    vector=vector,
                    payload=payload,
                )
            )
            index += 1
        if QDRANT_CLIENT.collection_exists(collection_name=collection_name):
            QDRANT_CLIENT.delete_collection(collection_name=collection_name)
        QDRANT_CLIENT.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSION, distance=Distance.COSINE
            ),
        )
        QDRANT_CLIENT.upsert(
            collection_name=collection_name,
            points=points,
        )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("data", type=Path)

    return parser.parse_args()


def main(args):
    data = json.load(open(args.data, "r"))
    qdrant_store = defaultdict(lambda: defaultdict(tuple))

    for class_data in tqdm(data, desc=Fore.RED, position=0, leave=True):
        process_per_class(class_data, qdrant_store)

    try:
        with open("data/qdrant.pkl", "wb") as fp:
            pickle.dump(defaultdict_to_dict(qdrant_store), fp)
    except Exception as e:
        print(e)
        pass

    push_qdrant_store(qdrant_store, ["good_or_service"])


if __name__ == "__main__":
    init(autoreset=True)
    setup_logger()
    main(parse_args())
