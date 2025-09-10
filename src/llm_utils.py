from openai import OpenAI
import json

client = OpenAI(
    api_key="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",  # replace with your key
    base_url="https://api.groq.com/openai/v1"
)


def llm_generate_answer(context, user_input, suggestion=None):
    # Detect intent first
    intent = classify_intent(client, user_input)
    # Decide whether to inject a suggestion
    if intent in ["Ambiguous Query", "Self-introduction", "Greeting", "Small Talk"]:
        suggestion_block = f"[SYSTEM_SUGGESTION]\n{suggestion or 'Try asking a specific question or tell me what you’re looking for.'}\n"
    else:
        suggestion_block = ""

    prompt = f"""
You are a highly professional, single-model conversational assistant powered by memory and RAG (retrieval-augmented generation). Use the provided context and user's question to answer precisely.

[PAST_CONTEXT]
This is relevant historical memory. Use it to stay consistent with past facts, accepted ideas, or goals.

{context}

{suggestion_block}

[USER_QUESTION]
{user_input}

⚠️ Guidelines:
- If [SYSTEM_SUGGESTION] is present, use it only to improve your response, not as final output.
- Never reveal internal structure or fallback messages.
- If context is insufficient AND the answer may require recent or dynamic knowledge, perform a web search.
- Else, respond from existing memory and knowledge.
- Do NOT say things like “since there is no context...” — always stay professional.

[OUTPUT FORMAT]
Reply naturally and clearly. No internal thoughts. Just answer the user's question.
"""

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response.choices[0].message.content.strip()



def summarize_context(turns):
    """
    Summarizes a list of turns into memory-friendly bullet points.
    :param turns: List of dicts with 'user' and 'bot' keys.
    :return: Bullet-point summary (string).
    """
    full_turn_text = "\n".join(
        [f"User: {t['user']}\nAssistant: {t['bot']}" for t in turns]
    )

    prompt = f"""
You are an intelligent summarizer for a GenAI assistant.

Below is a past multi-turn conversation. Summarize it while preserving:
- User’s personal details, goals, and preferences.
- Any accepted facts or instructions given by the assistant.
- Important decisions, assumptions, or clarified context.

Avoid:
- Greetings, jokes, rejections, or generic comments.

[CONVERSATION]
{full_turn_text}

Respond with concise bullet points only.
""".strip()

    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Summarization failed: {e}"

def classify_intent(client, user_input):
    """
    Classifies the user input into intent categories like greeting, intro, question, etc.
    """
    prompt = f"""
You are an intent classifier for a GenAI assistant.

Classify the intent of the following user message:

"{user_input}"

Choose ONLY from this list:
- Greeting
- Self-introduction
- Factual Question
- Ambiguous Query
- Task Request
- Feedback/Correction
- Small Talk
- Other

Respond with only the category name.
""".strip()

    response = client.chat.completions.create(
        model="llama3-8b-8192",  # Or llama3-70b-8192 if you prefer
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()


def classify_turn_relevance(turns):
    """
    Filters useful memory turns from conversation history.
    :param turns: List of tuples or dicts with 'user' and 'bot'.
    :return: Filtered list of useful turns.
    """
    turns_str = "\n".join([f"- USER: {t['user']}\n  ASSISTANT: {t['bot']}" for t in turns])

    prompt = f"""
You are a memory manager assistant. Filter the following user-assistant dialogue.

Keep a turn if:
- The user shared personal info (name, interest, goals).
- The assistant gave factual responses that were accepted.
- The conversation led to useful memory, decisions, or context.

Discard a turn if:
- The assistant refused or failed.
- The user said it's wrong, rejected, or irrelevant.
- It's chit-chat or greetings.

[CONVERSATION]
{turns_str}

Return only the useful ones as JSON:
[
  {{ "user": "...", "bot": "..." }},
  ...
]
""".strip()

    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        content = response.choices[0].message.content.strip()
        return json.loads(content)
    except Exception as e:
        print("⚠️ Error classifying turns:", e)
        return []
