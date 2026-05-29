"""MCP server — exposes the UX research repository as tools.

Allows Claude Desktop, ChatGPT, or Cursor to query interview transcripts
directly via the Model Context Protocol.

Run locally:
    python ux_mcp_server.py

Test with MCP Inspector:
    npx @modelcontextprotocol/inspector python ux_mcp_server.py

Add to Claude Desktop (claude_desktop_config.json):
    {
      "mcpServers": {
        "ux-research": {
          "command": "python",
          "args": ["/path/to/ux_mcp_server.py"]
        }
      }
    }
"""

import json
import os
from pathlib import Path

import chromadb
from groq import Groq
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from ingest import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    ingest_sample_sessions,
    retrieve,
)

# ---------------------------------------------------------------------------
# Bootstrap: load embedding model + ChromaDB
# ---------------------------------------------------------------------------

_model      = SentenceTransformer(EMBEDDING_MODEL)
_chroma     = chromadb.PersistentClient(path=CHROMA_DIR)
_collection = _chroma.get_or_create_collection(
    name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
)

SAMPLE_JSON = str(Path(__file__).parent / "data" / "sample_sessions.json")

# Auto-ingest sample data if collection is empty
if _collection.count() == 0 and os.path.exists(SAMPLE_JSON):
    print("Ingesting sample sessions into ChromaDB…")
    ingest_sample_sessions(SAMPLE_JSON, _collection, _model)
    print(f"Loaded {_collection.count()} chunks.")


# ---------------------------------------------------------------------------
# Pydantic schemas for tool arguments
# ---------------------------------------------------------------------------

class SearchArgs(BaseModel):
    query: str  = Field(description="Natural language search query over interview transcripts")
    top_k: int  = Field(default=5, ge=1, le=10, description="Number of results to return")


class SummaryArgs(BaseModel):
    focus: str = Field(
        default="overall pain points and recurring themes",
        description="What aspect to summarize (e.g. 'onboarding friction', 'navigation issues')"
    )


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

server = Server("ux-research-intelligence")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_interview_transcripts",
            description=(
                "Semantic search over UX interview transcripts. "
                "Returns relevant excerpts with session ID, participant info, and similarity score. "
                "Use this to find specific quotes, pain points, or themes across sessions."
            ),
            inputSchema=SearchArgs.model_json_schema(),
        ),
        types.Tool(
            name="generate_executive_summary",
            description=(
                "Synthesize an executive UX summary from the research repository. "
                "Uses the LLM to surface the top recurring themes and pain points, "
                "grounded in transcript evidence with session citations."
            ),
            inputSchema=SummaryArgs.model_json_schema(),
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:

    if name == "search_interview_transcripts":
        args   = SearchArgs(**arguments)
        chunks = retrieve(args.query, _model, _collection, top_k=args.top_k)
        results = [
            {
                "session":    c["source"],
                "segment":    c["page"],
                "similarity": c["score"],
                "excerpt":    c["text"],
            }
            for c in chunks
        ]
        return [types.TextContent(type="text", text=json.dumps(results, indent=2))]

    if name == "generate_executive_summary":
        args   = SummaryArgs(**arguments)
        # Retrieve broad context across the corpus
        chunks = retrieve(args.focus, _model, _collection, top_k=8)
        excerpts = "\n\n".join(
            f"[{c['source']}]\n{c['text']}" for c in chunks
        )
        prompt = (
            f"You are a UX research analyst. Based only on the following interview excerpts, "
            f"write an executive summary focused on: {args.focus}\n\n"
            f"For each theme, cite the session ID in brackets.\n\n"
            f"Excerpts:\n{excerpts}"
        )
        client   = Groq()
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_completion_tokens=1024,
        )
        summary = response.choices[0].message.content
        return [types.TextContent(type="text", text=summary)]

    raise ValueError(f"Unknown tool: {name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
