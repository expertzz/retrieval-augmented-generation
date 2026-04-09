import os
import sys
from ingest import ingest
from embed import embed_chunks
from store import build
from generate import generate


def setup(pdf_path: str) -> None:
    print(f"Ingesting '{pdf_path}'...")
    chunks = ingest(pdf_path)
    print(f"  {len(chunks)} chunks extracted")

    print("Embedding chunks...")
    chunks = embed_chunks(chunks)

    print("Building index...")
    build(chunks)
    print("  Ready\n")


def main() -> None:
    index_exists = os.path.exists("index.faiss") and os.path.exists("chunks.json")

    if len(sys.argv) > 1:
        setup(sys.argv[1])
    elif not index_exists:
        print("Usage: python3 main.py <path-to-pdf>")
        sys.exit(1)

    print("Ask a question (or type 'exit' to quit):\n")
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query:
            continue
        if query.lower() == "exit":
            break

        print(generate(query))
        print()


if __name__ == "__main__":
    main()
