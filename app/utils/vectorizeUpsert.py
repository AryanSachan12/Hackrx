"""
Pinecone Vector Store Integration
Handles vectorization and upserting of document chunks to Pinecone for semantic search.
"""

import os
import uuid
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import numpy as np

# Pinecone imports
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# Import our document processor
from .extractChunks import DocumentProcessor

# Load environment variables
load_dotenv()


class PineconeVectorStore:
    """Handles Pinecone vector operations for document chunks"""
    
    def __init__(self, 
                 index_name: str = "document-chunks",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 dimension: int = 384,
                 metric: str = "cosine"):
        """
        Initialize Pinecone vector store
        
        Args:
            index_name: Name of the Pinecone index
            embedding_model: Sentence transformer model name
            dimension: Vector dimension (must match embedding model)
            metric: Distance metric for similarity search
        """
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        
        # Initialize Pinecone
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        self.pc = Pinecone(api_key=api_key)
        
        # Initialize embedding model
        print(f"🤖 Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        print(f"✅ Embedding model loaded (dimension: {self.embedding_model.get_sentence_embedding_dimension()})")
        
        # Create or connect to index
        self._setup_index()
    
    def _setup_index(self):
        """Create or connect to Pinecone index"""
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                print(f"📊 Creating new Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                print(f"✅ Index '{self.index_name}' created successfully")
            else:
                print(f"📊 Connecting to existing index: {self.index_name}")
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            
            # Get index stats
            stats = self.index.describe_index_stats()
            print(f"📈 Index stats: {stats.total_vector_count} vectors, {stats.dimension} dimensions")
            
        except Exception as e:
            print(f"❌ Error setting up Pinecone index: {str(e)}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            print(f"🔄 Generating embeddings for {len(texts)} texts...")
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            print(f"✅ Generated {len(embeddings)} embeddings")
            return embeddings.tolist()
        except Exception as e:
            print(f"❌ Error generating embeddings: {str(e)}")
            raise
    
    def validate_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        Validate that chunks are properly formatted
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            True if chunks are valid, False otherwise
        """
        if not chunks:
            print("❌ No chunks provided for validation")
            return False
        
        print(f"🔍 Validating {len(chunks)} chunks...")
        
        for i, chunk in enumerate(chunks):
            if 'text' not in chunk:
                print(f"❌ Chunk {i} missing 'text' field")
                return False
            
            if 'chunk_index' not in chunk:
                print(f"❌ Chunk {i} missing 'chunk_index' field")
                return False
            
            if 'metadata' not in chunk:
                print(f"❌ Chunk {i} missing 'metadata' field")
                return False
            
            if not chunk['text'].strip():
                print(f"⚠️ Chunk {i} has empty text content")
        
        print(f"✅ All {len(chunks)} chunks validated successfully")
        return True
    
    def upsert_chunks(self, chunks: List[Dict[str, Any]], namespace: str = "") -> Dict[str, Any]:
        """
        Upsert document chunks to Pinecone
        
        Args:
            chunks: List of chunk dictionaries from document processor
            namespace: Optional namespace for organizing vectors
            
        Returns:
            Upsert statistics
        """
        if not chunks:
            print("⚠️ No chunks to upsert")
            return {"upserted_count": 0}
        
        # Validate chunks before processing
        if not self.validate_chunks(chunks):
            print("❌ Chunk validation failed")
            return {"upserted_count": 0, "error": "Invalid chunks"}
        
        try:
            print(f"🚀 Preparing to upsert {len(chunks)} chunks to Pinecone...")
            print(f"📝 Debug: First chunk preview: {chunks[0]['text'][:100]}..." if chunks else "No chunks")
            print(f"📝 Debug: Last chunk preview: {chunks[-1]['text'][:100]}..." if chunks else "No chunks")
            
            # Extract texts for embedding
            texts = [chunk['text'] for chunk in chunks]
            
            # Generate embeddings
            embeddings = self.generate_embeddings(texts)
            
            # Prepare vectors for upsert
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Create unique vector ID using source file + chunk index + UUID
                source_file = chunk['metadata'].get('source_file', 'unknown')
                chunk_index = chunk.get('chunk_index', i)
                vector_id = f"{source_file}_chunk_{chunk_index}_{uuid.uuid4().hex[:8]}"
                
                # Prepare metadata (Pinecone has metadata size limits)
                metadata = {
                    'text': chunk['text'][:1000],  # Truncate text for metadata
                    'chunk_index': chunk['chunk_index'],
                    'chunk_size': chunk['chunk_size'],
                    'source_file': chunk['metadata'].get('source_file', ''),
                    'file_extension': chunk['metadata'].get('file_extension', ''),
                    'has_tables': chunk['metadata'].get('has_tables', False),
                    'has_headings': chunk['metadata'].get('has_headings', False),
                    'contains_table': chunk['metadata'].get('contains_table', False),
                    'contains_heading': chunk['metadata'].get('contains_heading', False),
                    'is_first_chunk': chunk['metadata'].get('is_first_chunk', False),
                    'is_last_chunk': chunk['metadata'].get('is_last_chunk', False),
                    'total_chunks': chunk['metadata'].get('total_chunks', 0)
                }
                
                vectors.append({
                    'id': vector_id,
                    'values': embedding,
                    'metadata': metadata
                })
            
            print(f"📊 Debug: Created {len(vectors)} vectors from {len(chunks)} chunks")
            print(f"📝 Debug: First vector ID: {vectors[0]['id']}" if vectors else "No vectors")
            print(f"📝 Debug: Last vector ID: {vectors[-1]['id']}" if vectors else "No vectors")
            
            # Upsert in batches (Pinecone recommends batch size of 100)
            batch_size = 100
            total_upserted = 0
            
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                
                print(f"📤 Upserting batch {i//batch_size + 1}/{(len(vectors) + batch_size - 1)//batch_size} ({len(batch)} vectors)")
                
                response = self.index.upsert(
                    vectors=batch,
                    namespace=namespace
                )
                
                total_upserted += response.upserted_count
                print(f"✅ Batch upserted: {response.upserted_count} vectors")
            
            print(f"🎉 Successfully upserted {total_upserted} vectors to Pinecone!")
            
            return {
                "upserted_count": total_upserted,
                "total_chunks": len(chunks),
                "namespace": namespace,
                "index_name": self.index_name
            }
            
        except Exception as e:
            print(f"❌ Error upserting chunks: {str(e)}")
            raise
    
    def search_similar(self, 
                      query: str, 
                      top_k: int = 5, 
                      namespace: str = "",
                      include_metadata: bool = True) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using semantic similarity
        
        Args:
            query: Search query text
            top_k: Number of results to return
            namespace: Namespace to search in
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of similar chunks with scores
        """
        try:
            print(f"🔍 Searching for similar chunks: '{query[:50]}...'")
            print(f"📂 Namespace: '{namespace}'")
            print(f"🔢 Top K: {top_k}")
            
            # Generate embedding for query
            print("🧠 Generating query embedding...")
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            print(f"✅ Query embedding generated (dimension: {len(query_embedding)})")
            
            # Search in Pinecone
            print("🔎 Querying Pinecone index...")
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=namespace,
                include_metadata=include_metadata
            )
            
            print(f"📊 Pinecone query response: {len(results.matches)} matches")
            
            # Format results
            similar_chunks = []
            for i, match in enumerate(results.matches):
                chunk_data = {
                    'id': match.id,
                    'score': match.score,
                    'metadata': match.metadata if include_metadata else None
                }
                similar_chunks.append(chunk_data)
                print(f"  {i+1}. ID: {match.id}, Score: {match.score:.4f}")
                if include_metadata and match.metadata:
                    text_preview = match.metadata.get('text', '')[:100]
                    print(f"      Text preview: {text_preview}...")
            
            print(f"✅ Found {len(similar_chunks)} similar chunks")
            return similar_chunks
            
        except Exception as e:
            print(f"❌ Error searching similar chunks: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get current index statistics"""
        try:
            stats = self.index.describe_index_stats()
            return {
                'total_vectors': stats.total_vector_count,
                'dimension': stats.dimension,
                'index_fullness': stats.index_fullness,
                'namespaces': stats.namespaces
            }
        except Exception as e:
            print(f"❌ Error getting index stats: {str(e)}")
            return {}
    
    def delete_namespace(self, namespace: str):
        """Delete all vectors in a specific namespace"""
        try:
            print(f"🗑️ Deleting namespace: {namespace}")
            self.index.delete(delete_all=True, namespace=namespace)
            print(f"✅ Namespace '{namespace}' deleted")
        except Exception as e:
            print(f"❌ Error deleting namespace: {str(e)}")
            raise


class DocumentVectorizer:
    """Main class that combines document processing and vectorization"""
    
    def __init__(self, 
                 index_name: str = "document-chunks",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize document vectorizer
        
        Args:
            index_name: Pinecone index name
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            embedding_model: Sentence transformer model
        """
        print("🚀 Initializing Document Vectorizer...")
        
        # Initialize document processor
        self.doc_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Initialize vector store
        self.vector_store = PineconeVectorStore(
            index_name=index_name,
            embedding_model=embedding_model
        )
        
        print("✅ Document Vectorizer ready!")
    
    def process_and_upsert(self, 
                          file_path: str, 
                          namespace: str = "",
                          cleanup: bool = True) -> Dict[str, Any]:
        """
        Process a document and upsert its chunks to Pinecone
        
        Args:
            file_path: Path to document file or URL
            namespace: Pinecone namespace for organization
            cleanup: Whether to cleanup downloaded files
            
        Returns:
            Combined processing and upsert results
        """
        try:
            print(f"\n{'='*60}")
            print(f"🔄 Processing and vectorizing: {file_path}")
            print(f"{'='*60}")
            
            # Step 1: Process document and extract chunks
            result = self.doc_processor.process_file(file_path, cleanup=cleanup)
            
            if not result['success']:
                print(f"❌ Document processing failed: {result.get('error', 'Unknown error')}")
                return result
            
            chunks = result['chunks']
            if not chunks:
                print("⚠️ No chunks extracted from document")
                result['vectorization'] = {"upserted_count": 0}
                return result
            
            # Step 2: Upsert chunks to Pinecone
            print(f"\n📤 Upserting {len(chunks)} chunks to Pinecone...")
            upsert_result = self.vector_store.upsert_chunks(chunks, namespace=namespace)
            
            # Step 3: Combine results
            result['vectorization'] = upsert_result
            result['vectorization']['namespace'] = namespace
            
            print(f"\n🎉 Complete! Document processed and vectorized successfully!")
            print(f"📊 Final Stats:")
            print(f"   📄 Document chunks: {len(chunks)}")
            print(f"   📤 Vectors upserted: {upsert_result['upserted_count']}")
            print(f"   🏷️ Namespace: {namespace or 'default'}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error in process_and_upsert: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'chunks': [],
                'vectorization': {'upserted_count': 0}
            }
    
    def process_multiple_and_upsert(self, 
                                   file_paths: List[str], 
                                   namespace: str = "",
                                   cleanup: bool = True) -> List[Dict[str, Any]]:
        """
        Process multiple documents and upsert to Pinecone
        
        Args:
            file_paths: List of file paths or URLs
            namespace: Pinecone namespace
            cleanup: Whether to cleanup downloaded files
            
        Returns:
            List of processing results
        """
        results = []
        total_upserted = 0
        
        print(f"\n{'='*60}")
        print(f"📂 Batch Processing {len(file_paths)} documents...")
        print(f"{'='*60}")
        
        for i, file_path in enumerate(file_paths, 1):
            print(f"\n📄 Processing file {i}/{len(file_paths)}: {file_path}")
            
            result = self.process_and_upsert(file_path, namespace, cleanup)
            results.append(result)
            
            if result['success']:
                total_upserted += result['vectorization']['upserted_count']
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        print(f"\n{'='*60}")
        print(f"📈 Batch Processing Complete!")
        print(f"   ✅ Successful: {successful}/{len(file_paths)}")
        print(f"   📤 Total vectors upserted: {total_upserted}")
        print(f"   🏷️ Namespace: {namespace or 'default'}")
        print(f"{'='*60}")
        
        return results
    
    def search_documents(self, 
                        query: str, 
                        top_k: int = 5, 
                        namespace: str = "") -> List[Dict[str, Any]]:
        """
        Search for relevant document chunks
        
        Args:
            query: Search query
            top_k: Number of results to return
            namespace: Namespace to search in
            
        Returns:
            List of relevant chunks with similarity scores
        """
        return self.vector_store.search_similar(query, top_k, namespace)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vectorization statistics"""
        return self.vector_store.get_index_stats()


def test_chunk_processing():
    """Test function to debug chunk processing issues"""
    
    print("🔧 Testing Chunk Processing")
    print("=" * 50)
    
    try:
        # Create sample chunks to test
        sample_chunks = [
            {
                'text': f'This is sample chunk {i} with some content to test the vectorization process. Each chunk should be unique and properly upserted to Pinecone.',
                'chunk_index': i,
                'chunk_size': 80 + i * 5,
                'metadata': {
                    'chunk_id': f'test_chunk_{i}',
                    'source_file': 'test_document.txt',
                    'file_extension': '.txt',
                    'has_tables': False,
                    'has_headings': i == 0,
                    'contains_table': False,
                    'contains_heading': i == 0,
                    'is_first_chunk': i == 0,
                    'is_last_chunk': i == 4,
                    'total_chunks': 5
                }
            }
            for i in range(5)
        ]
        
        print(f"📊 Created {len(sample_chunks)} test chunks")
        for i, chunk in enumerate(sample_chunks):
            print(f"  Chunk {i}: {chunk['text'][:50]}...")
        
        # Initialize vector store
        vector_store = PineconeVectorStore(
            index_name="hackrx-test",
            embedding_model="all-MiniLM-L6-v2"
        )
        
        # Clear test namespace first
        try:
            vector_store.delete_namespace("test")
            print("🗑️ Cleared test namespace")
        except:
            print("⚠️ Test namespace was already empty or doesn't exist")
        
        # Test upserting
        result = vector_store.upsert_chunks(sample_chunks, namespace="test")
        
        print(f"🎉 Test completed! Result: {result}")
        
        # Test search
        if result['upserted_count'] > 0:
            search_results = vector_store.search_similar(
                "sample content test", 
                top_k=5, 
                namespace="test"
            )
            print(f"🔍 Search found {len(search_results)} results")
            for i, result in enumerate(search_results):
                print(f"   {i+1}. Score: {result['score']:.4f}, ID: {result['id']}")
                print(f"       Text: {result['metadata']['text'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":

    print("🚀 Pinecone Document Vectorization System")
    print("=" * 60)
    
    doc_vectorizer = DocumentVectorizer(
        index_name="document-chunks",
        chunk_size=1000,
        chunk_overlap=200,
        embedding_model="all-MiniLM-L6-v2"
    )

    # doc_path = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    # print(f"📄 Processing document: {doc_path}")
    # result = doc_vectorizer.process_and_upsert(doc_path, namespace="test-namespace")
    # print(f"📊 Processing result: {result}")
    # print("\n🎉 Document processed and vectorized successfully!")
    # print(f"📊 Final Stats:")
    # print(f"   📄 Document chunks: {len(result['chunks'])}")
    # print(f"   📤 Vectors upserted: {result['vectorization']['upserted_count']}")
    # print(f"   🏷️ Namespace: {result['vectorization']['namespace'] or 'default'}")
    # print("=" * 60)

    search_query = "What is the waiting period for pre-existing diseases (PED) to be covered?"
    print(f"🔍 Searching for: '{search_query}'")
    results = doc_vectorizer.search_documents(search_query, top_k=5, namespace="test-namespace")
    print(f"📊 Search results: {len(results)} chunks found")
    for i, res in enumerate(results, 1):
        print(f"  {i}. ID: {res['id']}, Score: {res['score']:.4f}, Metadata: {res['metadata']}")
    print("=" * 60)
