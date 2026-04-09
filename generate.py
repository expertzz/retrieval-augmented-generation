import os
import anthropic
from dotenv import load_dotenv
from retrieve import retrieve

load_dotenv()

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def generate(query: str, top_k: int = 5) -> str:
    chunks = retrieve(query, top_k)

    context = "\n\n".join(
        f"[Page {c['page']}]: {c['text']}" for c in chunks
    )

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        system="""You are the most competent assistant with 40 years of experience and countless recognition in your endeavors. 
        Answer the user's question using only the context that was provided by the user. 
        If the answer is not in the context, say so.""",
        messages=[
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            }
        ],
    )

    return response.content[0].text


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the document about?"
    print(generate(query))
