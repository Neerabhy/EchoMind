import sys, os
from src.memory_cache import get_recent_turns, add_turn, get_full_memory, clear_memory
from src.retriever_model import retrieve_context
from src.llm_utils import llm_generate_answer


MEMORY_FILE = "memory.json"

def format_recent_turns():
    return "\n\n".join(
        f"User: {t['user']}\nBot: {t['bot']}" for t in get_recent_turns()
    ).strip()

def cleanup_and_exit():
    
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)
    print("Assistant: Goodbye! Your session has ended.\n")
    sys.exit(0)

def handle_interrupt(signal_received, frame):
    print("\n[System] Keyboard interrupt received.")
    cleanup_and_exit()

def main_chat_loop():
    print("🧠 GenAI Chatbot with Smart Memory")
    print("Type your question below. Use /clear to reset memory, /exit to quit.\n")
    print("Assistant: Hello! How can I help you today?\n")

    while True:
        user_input = input("User: ").strip()
        if not user_input:
            continue

        command = user_input.lower()
        if command in ["/exit", "exit", "quit"]:
            print("Assistant: Goodbye! Your session has ended.\n")
            cleanup_and_exit()
        if command == "/clear":
            clear_memory()
            print("[System] Memory cleared.\n")
            continue

        # Step 1: Recent memory (summarized turns so far)
        recent_context = format_recent_turns()
        
        # Step 2: Main model checks if it can answer
        preliminary_response = llm_generate_answer(f"\n[PAST_CONTEXT]\n{recent_context}", user_input)

        # Step 3: Main model decides if more detail is needed
        if "INSUFFICIENT_CONTEXT" in preliminary_response:
            # Step 4: Ask second model (retriever) for older memory
            memory_result = retrieve_context(user_input, get_full_memory())
            
            context_block = ""
            if memory_result["type"] == "memory_context":
                context_block = f"\n[PAST_CONTEXT]\n{recent_context}\n\n[RETRIEVED_DETAIL]\n{memory_result['data']}"
            elif memory_result["type"] == "suggestion":
                context_block = f"\n[PAST_CONTEXT]\n{recent_context}\n\n[SYSTEM_SUGGESTION]\n{memory_result['data']}"
            else:
                context_block = f"\n[PAST_CONTEXT]\n{recent_context}"

            final_response = llm_generate_answer(context_block, user_input)
        else:
            final_response = preliminary_response

        add_turn(user_input, final_response)
        print(f"Assistant: {final_response}\n")

if __name__ == "__main__":
    main_chat_loop()
