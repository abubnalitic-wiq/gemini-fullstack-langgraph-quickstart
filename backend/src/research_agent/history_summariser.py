import logging
import re
from typing import Any, List, Optional, Tuple

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from research_agent.state import OverallState


# ===== Get message details =====
def get_message_content(msg) -> Optional[str]:
    """Extract content from various message formats."""
    if isinstance(msg, dict):
        return msg.get("content", "")
    elif hasattr(msg, "content"):
        return msg.content
    return None


def get_message_role(msg) -> Optional[str]:
    """Extract role from various message formats."""
    if isinstance(msg, dict):
        return msg.get("role")
    elif isinstance(msg, HumanMessage):
        return "user"
    elif isinstance(msg, AIMessage):
        return "assistant"
    elif isinstance(msg, SystemMessage):
        return "system"
    elif isinstance(msg, ToolMessage):
        return "tool"
    elif hasattr(msg, "role"):
        return msg.role
    return None


def get_latest_user_message(messages: list) -> Optional[str]:
    """Extract the most recent user message from the message list.

    Args:
        messages: List of messages in either dict or LangChain message format

    Returns:
        The content of the latest user message, or None if not found
    """
    for msg in reversed(messages):
        if get_message_role(msg) == "user":
            return get_message_content(msg)
    return None


def get_all_user_messages(messages: list) -> List[str]:
    """Extract all user messages from the message list."""
    user_messages = []
    for msg in messages:
        if get_message_role(msg) == "user":
            content = get_message_content(msg)
            if content:
                user_messages.append(content)
    return user_messages


# ===== History Management Utilities =====


def count_gemini_tokens(text: str) -> int:
    """Estimate token count for Gemini models using average word-to-token ratio."""
    # Gemini models typically have ~1.3 words per token (empirical estimate)
    words = len(re.findall(r"\w+", text))
    avg_word_per_token = 1.3
    return int(words / avg_word_per_token)


def get_message_tokens(msg: BaseMessage) -> int:
    """Get token count for a message."""
    content = get_message_content(msg)
    role = get_message_role(msg)
    # Rough estimate including role tokens
    return count_gemini_tokens(f"{role}: {content}")


def truncate_messages(
    messages: List[BaseMessage],
    max_tokens: int = 4000,
    keep_first: int = 1,
    keep_last: int = 2,
) -> Tuple[List[BaseMessage], bool]:
    """Truncate messages to fit within token limit.

    Args:
        messages: List of messages to truncate
        max_tokens: Maximum token count allowed
        keep_first: Number of messages to always keep from start
        keep_last: Number of messages to always keep from end

    Returns:
        Tuple of (truncated messages, was_truncated boolean)
    """
    if len(messages) <= keep_first + keep_last:
        return messages, False

    # Calculate total tokens
    total_tokens = sum(get_message_tokens(msg) for msg in messages)

    if total_tokens <= max_tokens:
        return messages, False

    # Keep first and last messages
    kept_messages = messages[:keep_first] + messages[-keep_last:]
    kept_tokens = sum(get_message_tokens(msg) for msg in kept_messages)

    # Try to add messages from the middle
    middle_messages = messages[keep_first:-keep_last]
    added_messages = []

    for msg in reversed(middle_messages):  # Start from most recent
        msg_tokens = get_message_tokens(msg)
        if kept_tokens + msg_tokens <= max_tokens:
            added_messages.insert(0, msg)
            kept_tokens += msg_tokens
        else:
            break

    # Reconstruct message list
    final_messages = messages[:keep_first] + added_messages + messages[-keep_last:]

    return final_messages, True


def summarize_conversation(
    messages: List[BaseMessage],
    llm: ChatGoogleGenerativeAI,
    max_summary_tokens: int = 2000,
) -> str:
    """Summarize a conversation into a concise context.

    Args:
        messages: Messages to summarize
        llm: Language model to use for summarization
        max_summary_tokens: Maximum tokens for the summary

    Returns:
        Summary string
    """
    # Convert messages to readable format
    conversation = []
    for msg in messages:
        role = get_message_role(msg)
        content = get_message_content(msg)
        if role and content:
            conversation.append(f"{role.capitalize()}: {content}")

    conversation_text = "\n\n".join(conversation)

    summarization_prompt = f"""Summarize the following conversation into a concise context that preserves:
1. The main topics discussed
2. Key questions asked
3. Important findings or conclusions
4. Any specific data points or metrics mentioned

Keep the summary under {max_summary_tokens} tokens.

Conversation:
{conversation_text}

Summary:"""

    response = llm.invoke([HumanMessage(content=summarization_prompt)])
    return response.content


def create_history_context(
    messages: List[BaseMessage],
    llm: Optional[ChatGoogleGenerativeAI] = None,
    max_history_tokens: int = 2000,
    summarize_threshold: int = 3000,
) -> str:
    """Create a history context string from messages.

    Args:
        messages: All messages in the conversation
        llm: LLM for summarization (if needed)
        max_history_tokens: Maximum tokens for history context
        summarize_threshold: Token count that triggers summarization

    Returns:
        History context string
    """
    if not messages:
        return ""

    # Count total tokens
    total_tokens = sum(get_message_tokens(msg) for msg in messages)

    # If within limits, return as is
    if total_tokens <= max_history_tokens:
        history_parts = []
        for msg in messages:
            role = get_message_role(msg)
            content = get_message_content(msg)
            if role and content:
                history_parts.append(f"{role.capitalize()}: {content}")
        return "\n\n".join(history_parts)

    # If over threshold and LLM available, summarize older messages
    if total_tokens > summarize_threshold and llm:
        # Find split point (keep recent messages detailed)
        recent_tokens = 0
        split_idx = len(messages)

        for i in range(len(messages) - 1, -1, -1):
            msg_tokens = get_message_tokens(messages[i])
            if recent_tokens + msg_tokens > max_history_tokens // 2:
                split_idx = i + 1
                break
            recent_tokens += msg_tokens

        # Summarize older messages
        if split_idx < len(messages):
            older_messages = messages[:split_idx]
            recent_messages = messages[split_idx:]

            summary = summarize_conversation(
                older_messages, llm, max_history_tokens // 2
            )

            # Combine summary with recent messages
            history_parts = [f"Previous Context Summary: {summary}"]

            for msg in recent_messages:
                role = get_message_role(msg)
                content = get_message_content(msg)
                if role and content:
                    history_parts.append(f"{role.capitalize()}: {content}")

            return "\n\n".join(history_parts)

    # Otherwise, truncate
    truncated_messages, _ = truncate_messages(messages, max_history_tokens)

    history_parts = []
    for msg in truncated_messages:
        role = get_message_role(msg)
        content = get_message_content(msg)
        if role and content:
            history_parts.append(f"{role.capitalize()}: {content}")

    return "\n\n".join(history_parts)


# ===== State Updates =====


def add_to_state(state: OverallState) -> OverallState:
    """Add conversation_history field to state if not present."""
    if "conversation_history" not in state:
        state["conversation_history"] = []
    return state
