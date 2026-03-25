"""
whats-my-tune · core agent
Iterative, multi-modal music memory reconstruction using Claude as the reasoning engine.
"""

import anthropic
import json
from typing import Optional
from session import Session
from tools import TOOL_DEFINITIONS, dispatch_tool

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"

# ── System prompt ──────────────────────────────────────────────────────────────
# Following Anthropic best practices:
#   - Explicit role + task context first
#   - XML tags for structural clarity
#   - Positive + negative examples inline
#   - Tool use guidance with when/when-not-to patterns
#   - Adaptive thinking hint for multi-step reasoning turns
SYSTEM_PROMPT = """
<role>
You are Mnemo, a music memory reconstruction agent. Your job is NOT to search databases —
it is to help the user RECONSTRUCT a half-remembered song from incomplete, noisy,
cross-modal clues: hummed fragments, lyric snippets, emotional memories, decade context,
instrument descriptions, or visual associations.
</role>

<goal>
Maintain a running belief state — a ranked probability distribution over candidate songs —
and iteratively narrow it by eliciting the most discriminative clue at each turn.
Think of yourself as a detective, not a search engine.
</goal>

<session_state_format>
You will receive the current session state as JSON in <session_state> tags.
It contains:
  - candidates: list of (song, artist, confidence 0-1, evidence_so_far)
  - clues_collected: list of all clues provided so far
  - turn: current turn number
  - last_audio_analysis: result from the most recent audio tool call (if any)
</session_state_format>

<tools_guidance>
Use tools in this priority order:
1. analyse_audio     — ALWAYS call this first when audio_path is present in the message
2. search_melody     — call when you have enough pitch/rhythm info to query
3. search_lyrics     — call when user provides lyric fragments (even 2-3 words)
4. search_context    — call when decade, genre, mood, or cultural context is described
5. play_candidate    — call to play a short preview to the user for confirmation

Do NOT call search_melody and search_lyrics simultaneously on the first turn —
gather audio analysis first, then search.

Do NOT call play_candidate unless confidence > 0.7 for the top candidate.
</tools_guidance>

<reasoning_strategy>
After each new clue or tool result:
1. Update your internal belief state — which songs become more or less likely?
2. Identify the single most discriminative question that would halve your uncertainty.
3. Ask only ONE clarifying question per turn (not multiple).
4. Phrase questions as comparisons where possible: "Was it faster or slower than [reference]?"
   This is easier for users to answer than open-ended questions.

Think step by step before responding. Use <thinking> to reason about candidate updates
before producing your final response.
</reasoning_strategy>

<response_format>
Always respond with a JSON object containing:
{
  "updated_candidates": [...],  // ranked list, highest confidence first
  "next_question": "...",       // ONE discriminative question, or null if confident
  "message_to_user": "...",     // friendly, encouraging, conversational
  "confidence_summary": "..."   // e.g. "I'm 70% sure this is X by Y"
}
Do not include any text outside the JSON object.
</response_format>

<examples>
<example type="good_narrowing_question">
User has hummed something with a rising 4-note motif. Good question:
"Was the song mostly instrumental, or did it have prominent vocals?"
Bad question (too broad): "What genre was it?"
Bad question (multiple): "Was it fast? Did it have drums? Was it from the 90s?"
</example>

<example type="good_candidate_update">
User confirms "yes, it had strings and felt triumphant" — update candidates:
- boost any orchestral/cinematic tracks already in list
- demote electronic/hip-hop candidates
- add new candidates: film scores, classical crossover, epic pop with strings
</example>
</examples>

<constraints>
- Never claim certainty above 0.95 unless user explicitly confirms a candidate.
- Never give up. If audio matching fails, pivot to contextual reconstruction.
- Keep message_to_user under 3 sentences. Be warm but efficient.
- If the user says "that's it!" or confirms, call play_candidate and mark session resolved.
</constraints>
"""


def run_agent(session: Session, user_message: str, audio_path: Optional[str] = None) -> dict:
    """
    One agent turn. Handles tool use loop internally.
    Returns the parsed JSON response from the agent.
    """
    # Build user content — text + optional audio reference
    user_content = user_message
    if audio_path:
        user_content = f"{user_message}\n<audio_path>{audio_path}</audio_path>"

    # Append to session history
    session.add_message("user", user_content)

    # Inject current belief state as context
    messages_with_state = _inject_session_state(session)

    # ── Agentic tool-use loop ──────────────────────────────────────────────────
    # Following Anthropic pattern: loop until stop_reason == "end_turn"
    max_iterations = 6  # guard against runaway tool loops
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages_with_state,
            # Hint adaptive thinking for multi-step reasoning turns
            # (beneficial when session has >2 clues and candidates list is contested)
            thinking={"type": "enabled", "budget_tokens": 1024} if session.turn > 2 else {"type": "disabled"},
        )

        # Append assistant turn to working messages
        messages_with_state.append({
            "role": "assistant",
            "content": response.content
        })

        # Check stop reason
        if response.stop_reason == "end_turn":
            break

        if response.stop_reason == "tool_use":
            # Execute all requested tool calls and collect results
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = dispatch_tool(block.name, block.input, session)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result)
                    })

            # Feed results back into the loop
            messages_with_state.append({
                "role": "user",
                "content": tool_results
            })

    # ── Parse final response ───────────────────────────────────────────────────
    final_text = _extract_text(response.content)
    try:
        agent_response = json.loads(final_text)
    except json.JSONDecodeError:
        # Graceful fallback — agent responded outside JSON format
        agent_response = {
            "updated_candidates": session.candidates,
            "next_question": None,
            "message_to_user": final_text,
            "confidence_summary": "Still thinking..."
        }

    # Persist updated state to session
    session.update_from_agent_response(agent_response)
    session.add_message("assistant", final_text)

    return agent_response


def _inject_session_state(session: Session) -> list:
    """Prepend current belief state to the message list as a system-adjacent user turn."""
    messages = list(session.messages)

    # Insert state as the first user message if this isn't turn 0
    if session.turn > 0 and len(messages) > 0:
        state_injection = {
            "role": "user",
            "content": f"<session_state>{json.dumps(session.to_dict(), indent=2)}</session_state>"
        }
        messages = [state_injection] + messages

    return messages


def _extract_text(content_blocks) -> str:
    """Extract plain text from response content blocks, skipping thinking blocks."""
    parts = []
    for block in content_blocks:
        if hasattr(block, "type") and block.type == "text":
            parts.append(block.text)
    return "\n".join(parts).strip()
