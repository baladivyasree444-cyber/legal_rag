
import os
import pickle
import numpy as np
import faiss

from google import genai
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -------------------------------------------------
# 1. DOCUMENT METADATA
# -------------------------------------------------

def get_document_metadata(file_name):

    document_id = file_name.split("_")[0]

    if document_id.startswith("EMP"):
        document_type = "Employment"

    elif document_id.startswith("NDA"):
        document_type = "NDA"

    elif document_id.startswith("SVC"):
        document_type = "Services"

    elif document_id.startswith("VND"):
        document_type = "Vendor"

    else:
        document_type = "Other"

    return document_id, document_type


# -------------------------------------------------
# 2. SETUP GEMINI
# -------------------------------------------------

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

client = genai.Client(api_key=api_key)


# -------------------------------------------------
# 3. CONFIGURE TEXT SPLITTER
# -------------------------------------------------

chunk_size = 800
chunk_overlap = 100

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", ". ", "; ", " ", ""],
    length_function=len
)


# -------------------------------------------------
# 4. LOAD DOCUMENTS AND CREATE CHUNKS
# -------------------------------------------------

data_dir = "documents"

chunks = []

for file_name in sorted(os.listdir(data_dir)):

    if file_name.endswith(".txt"):

        file_path = os.path.join(data_dir, file_name)

        with open(file_path, "r", encoding="utf-8") as f:
            document = f.read()

        print("\nDocument loaded:", file_name)

        document_id, document_type = get_document_metadata(
            file_name
        )

        document_chunks = text_splitter.split_text(document)

        for chunk in document_chunks:

            chunks.append({
                "text": chunk,
                "source": file_name,
                "document_id": document_id,
                "document_type": document_type
            })

        print("Chunks generated:", len(document_chunks))


print("\nTotal chunks:", len(chunks))


# -------------------------------------------------
# 5. GENERATE GEMINI EMBEDDINGS
# -------------------------------------------------

chunk_texts = [chunk["text"] for chunk in chunks]

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=chunk_texts
)

chunk_embeddings = np.array(
    [embedding.values for embedding in result.embeddings],
    dtype="float32"
)

print("\nEmbedding shape:", chunk_embeddings.shape)


# -------------------------------------------------
# 6. STORE EMBEDDINGS IN FAISS
# -------------------------------------------------

dimension = chunk_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(chunk_embeddings)

print("Vectors stored in FAISS:", index.ntotal)


# -------------------------------------------------
# 7. SAVE FAISS INDEX AND CHUNKS
# -------------------------------------------------

os.makedirs("models", exist_ok=True)

faiss.write_index(
    index,
    "models/legal_faiss.index"
)

with open("models/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)


print("\nFAISS index saved successfully!")
print("Chunk metadata saved successfully!")
print("Chunk size:", chunk_size)
print("Chunk overlap:", chunk_overlap)
