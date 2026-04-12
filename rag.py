import json
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def build_vector_store():
    with open("knowledge_base.json") as f:
        kb = json.load(f)

    chunks = []

    for plan in kb["plans"]:
        chunks.append(
            f"{plan['name']} plan costs ${plan['price_monthly']} per month. "
            f"Features {', '.join(plan['features'])}"
        )

    for policy in kb["policies"]:
        chunks.append(policy["detail"])

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    docs = splitter.create_documents(chunks)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    return FAISS.from_documents(docs, embeddings)


VECTOR_DB = build_vector_store()


def retrieve(query: str):
    docs = VECTOR_DB.similarity_search(query, k=3)
    return "\n".join([d.page_content for d in docs])