from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader
import os
from pymilvus import (
    connections,
    utility,
    CollectionSchema,
    FieldSchema,
    DataType,
    Collection,
)
from dotenv import load_dotenv

load_dotenv("../.env")

db = connections.connect(
    user="root", password="jarvis@admin", host="192.168.1.98", port="19530"
)

embeddingd_api = os.getenv("OPENAI_EMBEDINGS_API_KEY")
embeddings = OpenAIEmbeddings(
    openai_api_key=embeddingd_api, model="text-embedding-ada-002"
)


def delete_collection(name: str = "gptTeacher"):
    """Deletes the collection from the vector database"""
    utility.drop_collection(name)


def create_new_collection(collection_name: str = "gptTeacher") -> Collection:
    page_num = FieldSchema(
        name="page_num",
        dtype=DataType.INT64,
        is_primary=True,
        description="Page number of the PDF",
    )

    text = FieldSchema(
        name="text",
        dtype=DataType.VARCHAR,
        max_length=4096,
        description="Text extracted from the PDF",
    )

    ada_embedings = FieldSchema(
        name="ada_embedings",
        dtype=DataType.FLOAT_VECTOR,
        dim=1536,
        index=True,
        description="Embeddings of the text extracted from the PDF",
    )

    scheme = CollectionSchema(
        fields=[page_num, text, ada_embedings],
        description="Embeddings of the text extracted from the PDF",
        enable_dynamic_field=True,
    )

    collection = Collection(
        name=collection_name, schema=scheme, using="default", shards_num=2
    )

    collection.create_index(
        field_name="ada_embedings",
        index_params={"metric_type": "L2"},
        index_name="ada_embedings_index",
    )
    collection.load()

    return collection


def embed_pdf(file_path: os.path):
    """Embeds a PDF file into the vector database"""

    delete_collection(name="gptTeacher")

    collection = create_new_collection(collection_name="gptTeacher")

    pdf_loader = PyPDFLoader(file_path=file_path, extract_images=True)
    pdf_content = pdf_loader.load()

    # Insert data to Milvus
    for page in pdf_content:
        text = page.page_content
        page_no = page.metadata["page"]

        embedding = embeddings.embed_query(text)
        collection.insert(
            [{"page_num": page_no, "text": text, "ada_embedings": embedding}]
        )

    return True
