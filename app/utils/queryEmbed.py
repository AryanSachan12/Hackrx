"""
Query Processing and RAG Pipeline
Handles query formatting, vectorization, similarity search, and LLM integration with Gemini API.
"""

import os
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import google.generativeai as genai

# Import our vector store
from .vectorizeUpsert import PineconeVectorStore

# Load environment variables
load_dotenv()


class QueryProcessor:
    """Handles query processing, vectorization, and RAG pipeline with Gemini API"""

    def __init__(
        self,
        index_name: str = "document-chunks",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize Query Processor

        Args:
            index_name: Name of the Pinecone index
            embedding_model: Sentence transformer model name
        """
        # Initialize vector store
        self.vector_store = PineconeVectorStore(
            index_name=index_name, embedding_model=embedding_model
        )

        # Initialize Gemini API
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")

        print("✅ Query Processor initialized successfully")

    def format_query(self, query: str) -> str:
        """
        Format and clean the input query

        Args:
            query: Raw input query

        Returns:
            Formatted query string
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # Remove extra whitespace and normalize
        formatted_query = re.sub(r"\s+", " ", query.strip())

        # Ensure query ends with a question mark if it's a question
        question_words = [
            "what",
            "how",
            "when",
            "where",
            "why",
            "who",
            "which",
            "can",
            "is",
            "are",
            "do",
            "does",
            "did",
        ]
        if any(formatted_query.lower().startswith(word) for word in question_words):
            if not formatted_query.endswith("?"):
                formatted_query += "?"

        print(f"📝 Original query: {query}")
        print(f"✨ Formatted query: {formatted_query}")

        return formatted_query

    def search_similar_chunks(
        self,
        query: str,
        k: int = 5,
        namespace: str = "",
        similarity_threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Search for k most similar chunks in Pinecone

        Args:
            query: Search query
            k: Number of similar chunks to retrieve
            namespace: Pinecone namespace to search in
            similarity_threshold: Minimum similarity score threshold

        Returns:
            List of similar chunks with metadata
        """
        try:
            print(f"🔍 Searching for {k} most similar chunks...")
            print(f"📂 Namespace: '{namespace}'")
            print(f"📊 Similarity threshold: {similarity_threshold}")

            # Search using vector store
            results = self.vector_store.search_similar(
                query=query, top_k=k, namespace=namespace, include_metadata=True
            )

            print(f"🔎 Raw search results: {len(results)} chunks found")

            # Debug: Show all scores before filtering
            if results:
                print("📈 All scores from search:")
                for i, result in enumerate(results):
                    score = result.get("score", 0.0)
                    chunk_id = result.get("id", "N/A")
                    print(f"  {i+1}. ID: {chunk_id}, Score: {score:.4f}")
            else:
                print("⚠️ No results returned from Pinecone search!")

                # Check if index has any vectors in this namespace
                try:
                    index_stats = self.vector_store.get_index_stats()
                    print(f"📊 Index stats: {index_stats}")

                    if "namespaces" in index_stats:
                        namespaces = index_stats["namespaces"]
                        if namespace in namespaces:
                            ns_stats = namespaces[namespace]
                            print(
                                f"📂 Namespace '{namespace}' has {ns_stats.get('vector_count', 0)} vectors"
                            )
                        else:
                            print(f"❌ Namespace '{namespace}' not found in index!")
                            print(f"📋 Available namespaces: {list(namespaces.keys())}")
                    else:
                        print("⚠️ No namespace information available")
                except Exception as stats_error:
                    print(f"❌ Error getting index stats: {str(stats_error)}")

            # Filter by similarity threshold
            filtered_results = [
                result
                for result in results
                if result.get("score", 0.0) >= similarity_threshold
            ]

            print(
                f"📊 Found {len(results)} total results, {len(filtered_results)} above threshold ({similarity_threshold})"
            )

            # If no results pass threshold, show what we would get with lower threshold
            if not filtered_results and results:
                print("⚠️ No results above threshold! Showing what's available:")
                for result in results:
                    score = result.get("score", 0.0)
                    print(
                        f"   Score: {score:.4f} (below threshold {similarity_threshold})"
                    )

            return filtered_results

        except Exception as e:
            print(f"❌ Error searching similar chunks: {str(e)}")
            import traceback

            traceback.print_exc()
            raise

    def create_context_from_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Create context string from retrieved chunks

        Args:
            chunks: List of chunk dictionaries from similarity search

        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant context found."

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get("metadata", {})
            text = metadata.get("text", "")
            source_file = metadata.get("source_file", "Unknown")
            score = chunk.get("score", 0.0)

            context_part = f"""
            [Context {i}] (Relevance: {score:.3f})
            Source: {source_file}
            Content: {text}
            """
            context_parts.append(context_part.strip())

        return "\n\n".join(context_parts)

    def generate_prompt(self, query: str, context: str) -> str:
        """
        Generate prompt for Gemini API

        Args:
            query: User query
            context: Context from retrieved chunks

        Returns:
            Formatted prompt string
        """
        prompt = f"""
        You are a highly knowledgeable insurance advisor AI assistant. You are given excerpts from an insurance policy document and a user's question. Your task is to answer based strictly on the context, using concise and formal policy-style language.

        Context:
        {context}

        User Question:
        {query}

        Instructions:
        1. Answer ONLY using facts found in the context.
        2. Do NOT mention that the context was used.
        3. Do NOT quote or cite section numbers or context directly.
        4. If the context does not provide enough information, say: "The context does not contain sufficient information to answer this question."
        5. Use clear, policy-like language — e.g., "A grace period of thirty days is provided..."
        6. Be concise, factual, and complete.
        7. Use a single paragraph unless multiple distinct conditions or points must be listed.

        Answer:
        """

        return prompt.strip()

    def query_gemini(self, prompt: str) -> str:
        """
        Query Gemini API with the formatted prompt

        Args:
            prompt: Formatted prompt with context and query

        Returns:
            Generated response from Gemini
        """
        try:
            print("🤖 Querying Gemini API...")

            response = self.model.generate_content(prompt)

            if response and response.text:
                print("✅ Received response from Gemini API")
                return response.text.strip()
            else:
                print("⚠️ Empty response from Gemini API")
                return "I'm sorry, I couldn't generate a response. Please try again."

        except Exception as e:
            print(f"❌ Error querying Gemini API: {str(e)}")
            return f"Error generating response: {str(e)}"

    def process_query(
        self,
        query: str,
        k: int = 5,
        namespace: str = "",
        similarity_threshold: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Complete RAG pipeline: format query, search chunks, query LLM

        Args:
            query: Raw user query
            k: Number of similar chunks to retrieve
            namespace: Pinecone namespace to search in
            similarity_threshold: Minimum similarity score threshold

        Returns:
            Dictionary containing query results and metadata
        """
        try:
            # Step 1: Format query
            formatted_query = self.format_query(query)

            # Step 2: Search for similar chunks
            similar_chunks = self.search_similar_chunks(
                query=formatted_query,
                k=k,
                namespace=namespace,
                similarity_threshold=similarity_threshold,
            )

            # Step 3: Create context from chunks
            context = self.create_context_from_chunks(similar_chunks)

            # Step 4: Generate prompt
            prompt = self.generate_prompt(formatted_query, context)

            # Step 5: Query Gemini API
            answer = self.query_gemini(prompt)

            # Return comprehensive results
            return {
                "original_query": query,
                "formatted_query": formatted_query,
                "retrieved_chunks": len(similar_chunks),
                "context": context,
                "answer": answer,
                "chunks_metadata": [
                    {
                        "id": chunk.get("id"),
                        "score": chunk.get("score"),
                        "source_file": chunk.get("metadata", {}).get(
                            "source_file", "Unknown"
                        ),
                    }
                    for chunk in similar_chunks
                ],
            }

        except Exception as e:
            print(f"❌ Error in query processing pipeline: {str(e)}")
            return {
                "original_query": query,
                "error": str(e),
                "answer": "I'm sorry, there was an error processing your query. Please try again.",
            }


def main():
    """Example usage of QueryProcessor"""
    try:
        # Initialize processor
        processor = QueryProcessor()

        # Example queries
        test_queries = [
            "What is the waiting period for pre-existing diseases?",
            "How can I claim my insurance?",
            "What documents are required for policy application?",
        ]

        for query in test_queries:
            print(f"\n{'='*50}")
            print(f"Testing query: {query}")
            print("=" * 50)

            result = processor.process_query(query=query, k=3, similarity_threshold=0.6)

            print(f"\n🎯 Answer: {result['answer']}")
            print(f"📊 Retrieved {result.get('retrieved_chunks', 0)} chunks")

            if "chunks_metadata" in result:
                print("\n📋 Source files:")
                for chunk_meta in result["chunks_metadata"]:
                    print(
                        f"  - {chunk_meta['source_file']} (score: {chunk_meta['score']:.3f})"
                    )

    except Exception as e:
        print(f"❌ Error in main: {str(e)}")


if __name__ == "__main__":
    main()
