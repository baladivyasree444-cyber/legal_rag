
from rag_pipeline import embed_question, search_faiss, retrieve_chunks


# -------------------------------------------------
# 1. EVALUATION TEST CASES
# -------------------------------------------------

test_cases = [

    {
        "question": "What is the warranty period for VND-001?",
        "document_id": "VND-001",
        "expected_source": "VND-001",
        "expected_keywords": ["12 months"],
        "expected_relevant_chunks": 1
    },

    {
        "question": "What is the warranty period for VND-002?",
        "document_id": "VND-002",
        "expected_source": "VND-002",
        "expected_keywords": ["18 months"],
        "expected_relevant_chunks": 1
    },

    {
        "question": "How much insurance coverage is required in VND-001?",
        "document_id": "VND-001",
        "expected_source": "VND-001",
        "expected_keywords": ["USD 2 million"],
        "expected_relevant_chunks": 1
    },

    {
        "question": "How much insurance coverage is required in VND-002?",
        "document_id": "VND-002",
        "expected_source": "VND-002",
        "expected_keywords": ["USD 5 million"],
        "expected_relevant_chunks": 1
    },

    {
        "question": "What is the termination notice period in VND-001?",
        "document_id": "VND-001",
        "expected_source": "VND-001",
        "expected_keywords": ["60 days"],
        "expected_relevant_chunks": 1
    },

    {
        "question": "What is the termination notice period in VND-002?",
        "document_id": "VND-002",
        "expected_source": "VND-002",
        "expected_keywords": ["30 days"],
        "expected_relevant_chunks": 1
    },

    {
        "question": "How long does the vendor have to cure a material breach in VND-001?",
        "document_id": "VND-001",
        "expected_source": "VND-001",
        "expected_keywords": ["30 days"],
        "expected_relevant_chunks": 1
    },

    {
        "question": "How long does the vendor have to cure a material breach in VND-002?",
        "document_id": "VND-002",
        "expected_source": "VND-002",
        "expected_keywords": ["15 days"],
        "expected_relevant_chunks": 1
    }

]


# -------------------------------------------------
# 2. EVALUATION VARIABLES
# -------------------------------------------------

total_tests = len(test_cases)

passed_tests = 0

precision_scores = []
recall_scores = []

k = 5


print("\n" + "=" * 70)
print("LEGAL RAG RETRIEVAL EVALUATION")
print("=" * 70)


# -------------------------------------------------
# 3. RUN EACH TEST
# -------------------------------------------------

for test_number, test in enumerate(test_cases, 1):

    question = test["question"]

    document_id = test["document_id"]

    expected_source = test["expected_source"]

    expected_keywords = [
        keyword.lower()
        for keyword in test["expected_keywords"]
    ]

    expected_relevant_chunks = test["expected_relevant_chunks"]

    print(f"\nTEST {test_number}: {question}")
    print("-" * 70)

    # Step 1: Convert question into embedding

    question_embedding = embed_question(question)

    # Step 2: Retrieve relevant chunks

    distances, indices = search_faiss(
        question_embedding,
        k=k,
        document_type="Vendor",
        document_id=document_id
    )

    retrieved_chunks = retrieve_chunks(indices)

    # Step 3: Check whether the expected source was retrieved

    source_check = (
        len(retrieved_chunks) > 0
        and all(
            expected_source in chunk["source"]
            for chunk in retrieved_chunks
        )
    )

    # Step 4: Identify relevant chunks using expected keywords

    relevant_chunks = []

    for chunk in retrieved_chunks:

        chunk_text = chunk["text"].lower()

        if any(
            keyword in chunk_text
            for keyword in expected_keywords
        ):

            relevant_chunks.append(chunk)

    # Step 5: Calculate Precision@K

    if retrieved_chunks:

        precision_at_k = (
            len(relevant_chunks) / len(retrieved_chunks)
        )

    else:

        precision_at_k = 0.0

    # Step 6: Calculate Recall@K

    recall_at_k = min(
        len(relevant_chunks) / expected_relevant_chunks,
        1.0
    )

    # Step 7: Check whether the expected information was found

    content_check = len(relevant_chunks) > 0

    # Step 8: Overall test result

    test_passed = source_check and content_check

    if test_passed:

        passed_tests += 1

    precision_scores.append(precision_at_k)

    recall_scores.append(recall_at_k)

    # Step 9: Display test details

    sources = list(dict.fromkeys(
        chunk["source"]
        for chunk in retrieved_chunks
    ))

    print("Expected Source:", expected_source)

    print("Retrieved Sources:", sources)

    print("Distances:", distances)

    print(
        "Source Check:",
        "PASS" if source_check else "FAIL"
    )

    print(
        "Relevant Chunks:",
        len(relevant_chunks)
    )

    print(
        "Content Check:",
        "PASS" if content_check else "FAIL"
    )

    print(
        f"Precision@{k}:",
        round(precision_at_k, 2)
    )

    print(
        f"Recall@{k}:",
        round(recall_at_k, 2)
    )

    print(
        "Overall Result:",
        "PASS" if test_passed else "FAIL"
    )


# -------------------------------------------------
# 4. FINAL EVALUATION SUMMARY
# -------------------------------------------------

average_precision = (
    sum(precision_scores) / len(precision_scores)
)

average_recall = (
    sum(recall_scores) / len(recall_scores)
)

retrieval_accuracy = (
    passed_tests / total_tests
) * 100


print("\n" + "=" * 70)
print("FINAL EVALUATION SUMMARY")
print("=" * 70)

print("Total Tests:", total_tests)

print("Passed Tests:", passed_tests)

print("Failed Tests:", total_tests - passed_tests)

print(
    "Retrieval Accuracy:",
    round(retrieval_accuracy, 2),
    "%"
)

print(
    f"Average Precision@{k}:",
    round(average_precision, 2)
)

print(
    f"Average Recall@{k}:",
    round(average_recall, 2)
)

print("=" * 70)
