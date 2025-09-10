import json
import os

MEMORY_FILE = "memory.json"

def is_useful_turn(user_msg: str, bot_msg: str) -> bool:
    """Only store meaningful user-bot turns."""
    fallback_phrases = [
        "I don't have any information",
        "please provide more context",
        "clarify who you are",
        "I'm not sure",
        "Sorry, I didn’t understand",
    ]
    if any(phrase in bot_msg.lower() for phrase in fallback_phrases):
        return False
    if len(user_msg.strip()) < 4:  # too short
        return False
    return True


def _load_memory():
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_memory(turns):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(turns, f, indent=2)

def add_turn(user, bot):
    turns = _load_memory()
    if not is_useful_turn(user, bot):
        return
    turns.append({"user": user, "bot": bot})
    _save_memory(turns)

def get_full_memory():
    turns = _load_memory()
    return "\n\n".join([f"User: {t['user']}\nAssistant: {t['bot']}" for t in turns])

def get_recent_turns(limit=5):
    return _load_memory()[-limit:]

def clear_memory():
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)
