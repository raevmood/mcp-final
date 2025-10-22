"""
Data Ingestion Script for DeviceFinder.ai RAG Context Database
Loads text content and chunks it into ChromaDB for retrieval
"""
import chromadb
from datetime import datetime
from typing import List, Dict
import os

# Constants
PERSIST_DIRECTORY = "./chroma_rag_db"
COLLECTION_NAME = "chatbot_context"
CHUNK_SIZE = 500  # characters per chunk
OVERLAP = 50  # character overlap between chunks


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Full text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks in characters
    
    Returns:
        List of text chunks
    """
    print(f"→ Chunking text (size={chunk_size}, overlap={overlap})")
    
    if not text:
        print("✗ Empty text provided")
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
        
        start += (chunk_size - overlap)
    
    print(f"✓ Created {len(chunks)} chunks")
    return chunks


def ingest_text_file(file_path: str) -> int:
    """
    Ingest a single text file into ChromaDB.
    
    Args:
        file_path: Path to text file
    
    Returns:
        Number of chunks added
    """
    print("=" * 60)
    print(f"→ Ingesting file: {file_path}")
    
    try:
        # Read file
        if not os.path.exists(file_path):
            print(f"✗ File not found: {file_path}")
            return 0
        
        print(f"→ Reading file...")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        file_size = len(content)
        print(f"✓ File read successfully ({file_size} characters)")
        
        if not content.strip():
            print("✗ File is empty")
            return 0
        
        # Initialize ChromaDB
        print(f"→ Connecting to ChromaDB at {PERSIST_DIRECTORY}")
        try:
            client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
            collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"✓ Connected to collection '{COLLECTION_NAME}'")
        except Exception as db_error:
            print(f"✗ ChromaDB connection failed: {db_error}")
            return 0
        
        # Chunk the text
        chunks = chunk_text(content)
        
        if not chunks:
            print("✗ No chunks created")
            return 0
        
        # Prepare data for ChromaDB
        print(f"→ Preparing {len(chunks)} chunks for ingestion...")
        documents = []
        metadatas = []
        ids = []
        
        timestamp = datetime.utcnow().isoformat()
        file_name = os.path.basename(file_path)
        
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            
            metadata = {
                "source_file": file_name,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "indexed_at": timestamp,
                "chunk_size": len(chunk)
            }
            metadatas.append(metadata)
            
            # Create unique ID
            chunk_id = f"{file_name}_{i}_{timestamp}".replace(" ", "_").replace("/", "_")
            ids.append(chunk_id)
        
        # Add to ChromaDB
        print(f"→ Adding chunks to ChromaDB...")
        try:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"✓ Successfully added {len(chunks)} chunks to ChromaDB")
            
            # Verify ingestion
            total_docs = collection.count()
            print(f"✓ Total documents in collection: {total_docs}")
            
            return len(chunks)
        
        except Exception as add_error:
            print(f"✗ Failed to add chunks to ChromaDB: {add_error}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return 0
    
    except Exception as e:
        print(f"✗ Ingestion failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return 0
    
    finally:
        print("=" * 60)


def clear_collection() -> bool:
    """
    Clear all documents from the collection.
    Use with caution!
    
    Returns:
        True if successful, False otherwise
    """
    print("=" * 60)
    print("⚠️  WARNING: Clearing entire collection")
    print("=" * 60)
    
    try:
        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        
        # Try to get the collection
        try:
            collection = client.get_collection(name=COLLECTION_NAME)
            doc_count = collection.count()
            
            print(f"→ Found collection '{COLLECTION_NAME}' with {doc_count} documents")
            
            if doc_count == 0:
                print("→ Collection already empty")
                return True
            
            # Delete the collection
            print(f"→ Deleting collection...")
            client.delete_collection(name=COLLECTION_NAME)
            print(f"✓ Collection deleted successfully")
            
            # Recreate empty collection
            print(f"→ Recreating empty collection...")
            client.create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"✓ Empty collection created")
            
            return True
        
        except Exception as collection_error:
            print(f"✗ Collection operation failed: {collection_error}")
            return False
    
    except Exception as e:
        print(f"✗ Failed to clear collection: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False
    
    finally:
        print("=" * 60)


def view_collection_stats() -> Dict:
    """
    View statistics about the current collection.
    
    Returns:
        Dictionary with collection statistics
    """
    print("=" * 60)
    print("📊 Collection Statistics")
    print("=" * 60)
    
    try:
        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        
        try:
            collection = client.get_collection(name=COLLECTION_NAME)
            doc_count = collection.count()
            
            print(f"Collection Name: {COLLECTION_NAME}")
            print(f"Total Documents: {doc_count}")
            print(f"Persist Directory: {PERSIST_DIRECTORY}")
            
            if doc_count > 0:
                # Get sample documents
                sample = collection.get(limit=3)
                print(f"\nSample Documents:")
                for i, doc in enumerate(sample['documents'], 1):
                    preview = doc[:100].replace('\n', ' ')
                    print(f"  {i}. {preview}...")
            
            stats = {
                "collection_name": COLLECTION_NAME,
                "total_documents": doc_count,
                "persist_directory": PERSIST_DIRECTORY
            }
            
            return stats
        
        except Exception as collection_error:
            print(f"✗ Collection not found or error: {collection_error}")
            return {"error": str(collection_error)}
    
    except Exception as e:
        print(f"✗ Failed to get collection stats: {e}")
        return {"error": str(e)}
    
    finally:
        print("=" * 60)


if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 60)
    print("DeviceFinder.ai RAG Data Ingestion Script")
    print("=" * 60 + "\n")
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python ingest_data.py <text_file_path>  # Ingest a text file")
        print("  python ingest_data.py --clear           # Clear collection")
        print("  python ingest_data.py --stats           # View statistics")
        print("\nExample:")
        print("  python ingest_data.py knowledge_base.txt")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "--clear":
        # Clear collection
        confirm = input("⚠️  Are you sure you want to clear the entire collection? (yes/no): ")
        if confirm.lower() == "yes":
            success = clear_collection()
            if success:
                print("✓ Collection cleared successfully")
            else:
                print("✗ Failed to clear collection")
        else:
            print("→ Operation cancelled")
    
    elif command == "--stats":
        # View statistics
        stats = view_collection_stats()
    
    else:
        # Ingest file
        file_path = command
        chunks_added = ingest_text_file(file_path)
        
        if chunks_added > 0:
            print(f"\n✓ Ingestion completed successfully!")
            print(f"✓ Added {chunks_added} chunks to the knowledge base")
            print(f"\nNext steps:")
            print(f"  1. Start the MCP server: python mcp_server.py")
            print(f"  2. Test retrieval using the RAG client")
        else:
            print(f"\n✗ Ingestion failed. Please check the errors above.")