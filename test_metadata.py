
import pickle

with open("models/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print("Total chunks:", len(chunks))

print("\nFirst 5 chunks with metadata:")

for chunk in chunks[:5]:

    print("\nSource:", chunk["source"])
    print("Document ID:", chunk["document_id"])
    print("Document Type:", chunk["document_type"])
    print("Text:", chunk["text"][:100])
