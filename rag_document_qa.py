"""
Document Q&A (Retrieval-based)
Ask a question about a document; the program finds and returns the most
relevant chunk of text as the answer, using TF-IDF similarity search.

Note: this implements the "retrieval" half of RAG (Retrieval-Augmented
Generation). A full RAG system would pass the retrieved text to an AI model
to generate a natural-language answer; here, the most relevant chunk is
returned directly, which keeps the project free to run and focuses on the
retrieval mechanics themselves.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SAMPLE_DOCUMENT = """
Photosynthesis is the process by which green plants convert sunlight into chemical energy.
This process takes place primarily in the leaves, within structures called chloroplasts.
Chlorophyll, the green pigment in chloroplasts, absorbs sunlight and initiates the reaction.

Water is absorbed by the roots and transported to the leaves through the plant's vascular system.
Carbon dioxide enters the leaves through small pores called stomata.
These two ingredients, water and carbon dioxide, are combined using light energy to produce glucose and oxygen.

The oxygen produced during photosynthesis is released into the atmosphere as a byproduct.
This oxygen is essential for most life on Earth, including human and animal respiration.
Glucose, the other product, is used by the plant as an energy source or stored for later use.

Photosynthesis is affected by several factors including light intensity, temperature, and carbon dioxide concentration.
Higher light intensity generally increases the rate of photosynthesis, up to a certain point.
Extremely high temperatures can damage the enzymes involved in the process, slowing it down.
"""


def chunk_document(text, chunk_size=2):
    """Splits the document into chunks of a few sentences each,
    so retrieval can find a specific relevant passage rather than
    the whole document at once."""
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = ". ".join(sentences[i:i + chunk_size]) + "."
        chunks.append(chunk)
    return chunks


class DocumentQA:
    def __init__(self, document_text):
        self.chunks = chunk_document(document_text)
        self.vectorizer = TfidfVectorizer()
        self.chunk_vectors = self.vectorizer.fit_transform(self.chunks)

    def answer(self, question, top_n=1):
        """Finds the chunk(s) most similar to the question using cosine similarity."""
        question_vector = self.vectorizer.transform([question])
        similarities = cosine_similarity(question_vector, self.chunk_vectors)[0]

        # Get indices of the top N most similar chunks
        top_indices = similarities.argsort()[::-1][:top_n]

        results = []
        for idx in top_indices:
            results.append({
                "chunk": self.chunks[idx],
                "similarity": similarities[idx],
            })
        return results


def main():
    qa = DocumentQA(SAMPLE_DOCUMENT)

    print("Document Q&A System (retrieval-based)")
    print("Ask a question about photosynthesis (or 'quit' to exit)\n")

    while True:
        question = input("Your question: ")
        if question.lower() == "quit":
            break

        results = qa.answer(question, top_n=1)
        best = results[0]

        print(f"\nMost relevant passage (similarity: {best['similarity']:.2f}):")
        print(f"{best['chunk']}\n")


if __name__ == "__main__":
    main()
