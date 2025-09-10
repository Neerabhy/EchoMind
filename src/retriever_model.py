from sentence_transformers import SentenceTransformer, util
from src.memory_cache import get_full_memory
from src.llm_utils import llm_generate_answer
import difflib

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_context(user_query, memory_text=None):
    if memory_text is None:
        memory_text = get_full_memory()

    # Step 1: Chunk into turns (per interaction)
    memory_chunks = [chunk.strip() for chunk in memory_text.split("\n\n") if chunk.strip()]
    if not memory_chunks:
        return {"type": "no_match", "summary": "No memory to search.", "data": None}

    # Step 2: Encode query and memory
    query_emb = embedder.encode(user_query, convert_to_tensor=True)
    chunk_embs = embedder.encode(memory_chunks, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_emb, chunk_embs)[0]

    # Step 3: Scoring
    top_score = float(scores.max().item())
    top_idx = int(scores.argmax())
    top_chunk = memory_chunks[top_idx]

    # Step 4: Adjust sensitivity for short inputs like "what is my name"
    token_count = len(user_query.strip().split())
    score_threshold = 0.3 if token_count <= 5 else 0.4

    if top_score >= score_threshold:
        return {
            "type": "memory_context",
            "summary": f"Found relevant context (score: {top_score:.2f})",
            "data": top_chunk
        }

    # Step 5: Fallback using fuzzy match (for names or specific tokens)
    fuzzy_match = fuzzy_context_match(user_query, memory_chunks)
    if fuzzy_match:
        return {
            "type": "memory_context",
            "summary": f"Fuzzy match fallback.",
            "data": fuzzy_match
        }

    # Step 6: Fallback suggestion
    suggestion = get_secondary_suggestion(user_query)
    return {
        "type": "suggestion" if suggestion else "no_match",
        "summary": "Fallback suggestion from model 2" if suggestion else "No match.",
        "data": suggestion
    }

def fuzzy_context_match(query, memory_chunks):
    # Use fuzzy matching only if query is short (e.g., "what is my name")
    if len(query.split()) > 6:
        return None
    matches = difflib.get_close_matches(query, memory_chunks, n=1, cutoff=0.4)
    return matches[0] if matches else None

def get_secondary_suggestion(query):
    prompt = f"""
You are a suggestion engine for a memory-driven GenAI system.

Given this user query:
"{query}"

If this query requires more info not found in memory, suggest:
- Key topics or keywords the assistant should explore
- Clarifications or intent behind the query

Respond only if you can suggest something useful.
If unsure, return nothing.

Suggestion:
"""
    response = llm_generate_answer("", prompt)
    return response.strip() if "suggest" in response.lower() else None
