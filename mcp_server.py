"""
MCP RAG Server for DeviceFinder.ai Chatbot Context Retrieval
"""
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import chromadb
from chromadb.config import Settings
import json

app = FastAPI(
    title="DeviceFinder.ai RAG Context Server",
    description="MCP server providing RAG context retrieval for chatbot queries",
    version="1.0.0"
)

origins = [
    "https://raevmood.github.io/final-frontend",
    "https://final-project-yv26.onrender.com"
]

# CORS Configuration - Wildcard for MVP (configure in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # TODO: Replace with specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global ChromaDB client and collection
chroma_client = None
context_collection = None

# Constants
PERSIST_DIRECTORY = "./chroma_rag_db"
COLLECTION_NAME = "chatbot_context"
TOP_K_RESULTS = 3


class MCPRequest(BaseModel):
    """JSON-RPC 2.0 request format"""
    jsonrpc: str = "2.0"
    id: int = 1
    method: str
    params: Optional[Dict[str, Any]] = {}


class MCPResponse(BaseModel):
    """JSON-RPC 2.0 response format"""
    jsonrpc: str = "2.0"
    id: int
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


@app.on_event("startup")
async def startup_event():
    """Initialize ChromaDB on server startup"""
    global chroma_client, context_collection
    
    print("=" * 60)
    print("DeviceFinder.ai RAG Context Server - Starting Up")
    print("=" * 60)
    
    try:
        print(f"→ Initializing ChromaDB at {PERSIST_DIRECTORY}")
        chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        
        # Get or create collection
        print(f"→ Loading collection: {COLLECTION_NAME}")
        context_collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Check collection status
        doc_count = context_collection.count()
        print(f"✓ ChromaDB initialized successfully")
        print(f"✓ Collection '{COLLECTION_NAME}' loaded with {doc_count} documents")
        
        if doc_count == 0:
            print("⚠️  Warning: Collection is empty. Run ingest_data.py to populate.")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Failed to initialize ChromaDB: {e}")
        raise RuntimeError(f"ChromaDB initialization failed: {e}") from e


@app.get("/", summary="Server health check")
async def root():
    """Health check endpoint"""
    try:
        doc_count = context_collection.count() if context_collection else 0
        print(f"→ Health check requested. Collection has {doc_count} documents.")
        
        return {
            "status": "healthy",
            "service": "DeviceFinder.ai RAG Context Server",
            "version": "1.0.0",
            "mcp_server_active": True,
            "collection": COLLECTION_NAME,
            "documents_indexed": doc_count,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }


@app.post("/mcp/tools/call", summary="MCP JSON-RPC endpoint")
async def mcp_tools_call(request: MCPRequest):
    """
    Main MCP endpoint following JSON-RPC 2.0 protocol.
    Handles tool calls for context retrieval.
    """
    print("=" * 60)
    print(f"→ MCP Request received: {request.method}")
    print(f"→ Request ID: {request.id}")
    print(f"→ Params: {request.params}")
    
    try:
        # Validate JSON-RPC version
        if request.jsonrpc != "2.0":
            error_msg = f"Invalid JSON-RPC version: {request.jsonrpc}"
            print(f"✗ {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Route to appropriate handler
        if request.method == "tools/call":
            tool_name = request.params.get("name")
            arguments = request.params.get("arguments", {})
            
            print(f"→ Tool called: {tool_name}")
            print(f"→ Arguments: {arguments}")
            
            if tool_name == "retrieve_context":
                result = await retrieve_context_tool(arguments)
                
                response = MCPResponse(
                    jsonrpc="2.0",
                    id=request.id,
                    result=result
                )
                
                print(f"✓ MCP request completed successfully")
                print("=" * 60)
                return response.dict()
            
            else:
                error_msg = f"Unknown tool: {tool_name}"
                print(f"✗ {error_msg}")
                response = MCPResponse(
                    jsonrpc="2.0",
                    id=request.id,
                    error={"code": -32601, "message": error_msg}
                )
                print("=" * 60)
                return response.dict()
        
        else:
            error_msg = f"Unknown method: {request.method}"
            print(f"✗ {error_msg}")
            response = MCPResponse(
                jsonrpc="2.0",
                id=request.id,
                error={"code": -32601, "message": error_msg}
            )
            print("=" * 60)
            return response.dict()
    
    except HTTPException as he:
        print(f"✗ HTTP Exception: {he.detail}")
        print("=" * 60)
        raise he
    
    except Exception as e:
        error_msg = f"Internal server error: {str(e)}"
        print(f"✗ {error_msg}")
        print("=" * 60)
        
        response = MCPResponse(
            jsonrpc="2.0",
            id=request.id,
            error={"code": -32603, "message": error_msg}
        )
        return response.dict()


async def retrieve_context_tool(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieve relevant context chunks for a given query.
    Never fails - always returns a valid response with chunks_found count.
    
    Args:
        arguments: Dict containing 'query' key with user's question
    
    Returns:
        MCP-formatted result with context chunks (or empty list if none found)
    """
    print("→ Executing retrieve_context tool")
    
    try:
        # Extract query
        query = arguments.get("query")
        if not query:
            error_msg = "Missing required argument: 'query'"
            print(f"✗ {error_msg}")
            # Return empty result, not error
            return {
                "content": [{
                    "type": "text", 
                    "text": json.dumps({
                        "query": "",
                        "context_chunks": [],
                        "chunks_found": 0,
                        "warning": error_msg
                    })
                }]
            }
        
        print(f"→ Query: '{query}'")
        
        # Check if collection is initialized
        if not context_collection:
            error_msg = "ChromaDB collection not initialized"
            print(f"✗ {error_msg}")
            # Return empty result, not error
            return {
                "content": [{
                    "type": "text", 
                    "text": json.dumps({
                        "query": query,
                        "context_chunks": [],
                        "chunks_found": 0,
                        "warning": error_msg
                    })
                }]
            }
        
        # Check if collection has documents
        doc_count = context_collection.count()
        if doc_count == 0:
            warning_msg = "No documents in collection"
            print(f"⚠️  {warning_msg}")
            # Return empty result, not error - this is expected if DB is empty
            return {
                "content": [{
                    "type": "text", 
                    "text": json.dumps({
                        "query": query,
                        "context_chunks": [],
                        "chunks_found": 0,
                        "warning": warning_msg
                    })
                }]
            }
        
        print(f"→ Searching {doc_count} documents for relevant context...")
        
        # Query ChromaDB
        try:
            results = context_collection.query(
                query_texts=[query],
                n_results=min(TOP_K_RESULTS, doc_count)
            )
            
            print(f"→ ChromaDB query completed")
            
        except Exception as query_error:
            error_msg = f"ChromaDB query failed: {str(query_error)}"
            print(f"✗ {error_msg}")
            # Return empty result, don't fail
            return {
                "content": [{
                    "type": "text", 
                    "text": json.dumps({
                        "query": query,
                        "context_chunks": [],
                        "chunks_found": 0,
                        "warning": error_msg
                    })
                }]
            }
        
        # Format results
        if not results['ids'] or not results['ids'][0]:
            print("→ No results found for query")
            formatted_result = {
                "query": query,
                "context_chunks": [],
                "chunks_found": 0
            }
        else:
            context_chunks = []
            for i, doc_id in enumerate(results['ids'][0]):
                chunk = {
                    "chunk_id": doc_id,
                    "content": results['documents'][0][i],
                    "relevance_score": 1 - results['distances'][0][i]  # Convert distance to similarity
                }
                context_chunks.append(chunk)
            
            formatted_result = {
                "query": query,
                "context_chunks": context_chunks,
                "chunks_found": len(context_chunks)
            }
            
            print(f"✓ Retrieved {len(context_chunks)} context chunks")
            for idx, chunk in enumerate(context_chunks, 1):
                content_preview = chunk['content'][:100].replace('\n', ' ')
                print(f"  {idx}. Score: {chunk['relevance_score']:.3f} | {content_preview}...")
        
        # Always return successful MCP format with chunks_found count
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(formatted_result, ensure_ascii=False)
                }
            ]
        }
    
    except Exception as e:
        error_msg = f"Unexpected error in retrieve_context: {str(e)}"
        print(f"✗ {error_msg}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        
        # Even on unexpected errors, return empty result not error
        return {
            "content": [{
                "type": "text", 
                "text": json.dumps({
                    "query": arguments.get("query", ""),
                    "context_chunks": [],
                    "chunks_found": 0,
                    "warning": error_msg
                })
            }]
        }


# Direct API endpoint (alternative to MCP)
class DirectContextRequest(BaseModel):
    query: str = Field(..., description="User query for context retrieval")


@app.post("/retrieve_context", summary="Direct context retrieval (non-MCP)")
async def retrieve_context_direct(request: DirectContextRequest):
    """
    Direct API endpoint for context retrieval (non-MCP protocol).
    Useful for testing and simple integrations.
    """
    print("=" * 60)
    print(f"→ Direct API request: /retrieve_context")
    print(f"→ Query: '{request.query}'")
    
    try:
        result = await retrieve_context_tool({"query": request.query})
        
        # Extract the JSON from MCP format
        if result.get("content"):
            content_text = result["content"][0].get("text", "{}")
            parsed_result = json.loads(content_text)
            
            print(f"✓ Direct API request completed")
            print("=" * 60)
            return parsed_result
        else:
            print(f"✗ No content in result")
            print("=" * 60)
            raise HTTPException(status_code=500, detail="No content in response")
    
    except json.JSONDecodeError as je:
        error_msg = f"JSON decode error: {str(je)}"
        print(f"✗ {error_msg}")
        print("=" * 60)
        raise HTTPException(status_code=500, detail=error_msg)
    
    except Exception as e:
        error_msg = f"Error in direct API: {str(e)}"
        print(f"✗ {error_msg}")
        print("=" * 60)
        raise HTTPException(status_code=500, detail=error_msg)


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("Starting DeviceFinder.ai RAG Context Server")
    print("=" * 60 + "\n")
    uvicorn.run("mcp_server:app", host="0.0.0.0", port=8001, reload=True)