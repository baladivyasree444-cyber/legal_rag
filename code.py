if __name__ == "__main__":

    question = "What is the employee's annual bonus?"

    result = answer_question(question)

    print("\n" + "=" * 60)
    print("USER QUESTION")
    print("=" * 60)

    print(result["question"])

    print("\n" + "=" * 60)
    print("GENERATED ANSWER")
    print("=" * 60)

    print(result["answer"])

    print("\n" + "=" * 60)
    print("SOURCE DOCUMENTS")
    print("=" * 60)

    for source in result["sources"]:
        print(source)

    print("\n" + "=" * 60)
    print("RETRIEVED CHUNKS")
    print("=" * 60)

    for i, chunk in enumerate(result["retrieved_chunks"]):

        print(f"\nChunk {i+1}")
        print("Source:", chunk["source"])
        print("Content:", chunk["text"])


return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": retrieved_chunks,
        "distances": distances.tolist()
    }