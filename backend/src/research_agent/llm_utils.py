import logging
import os
from typing import List, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from research_agent.configuration import Configuration

logging.basicConfig(level=logging.DEBUG)


def get_latest_user_message(messages: list) -> Optional[str]:
    """Extract the most recent user message from the message list.

    Args:
        messages: List of messages in either dict or LangChain message format

    Returns:
        The content of the latest user message, or None if not found
    """
    for msg in reversed(messages):
        # Handle dict format
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "")
        # Handle LangChain HumanMessage objects
        elif isinstance(msg, HumanMessage):
            return msg.content
        # Handle other message objects with role attribute
        elif hasattr(msg, "role") and hasattr(msg, "content") and msg.role == "user":
            return msg.content
    return None


def create_llm(
    config: RunnableConfig,
    model_type: str = "query_generator_model",
    temperature: float = 0.5,
    **kwargs,
) -> ChatGoogleGenerativeAI:
    """Create a standardized LLM instance.

    Args:
        config: Configuration for the runnable
        model_type: The model type key from Configuration
        temperature: Temperature setting for the model
        **kwargs: Additional parameters for ChatGoogleGenerativeAI

    Returns:
        Configured ChatGoogleGenerativeAI instance
    """
    configurable = Configuration.from_runnable_config(config)
    model_name = getattr(configurable, model_type, configurable.query_generator_model)

    default_params = {
        "model": model_name,
        "temperature": temperature,
        "max_retries": 2,
        "api_key": os.getenv("GEMINI_API_KEY"),
    }
    default_params.update(kwargs)

    return ChatGoogleGenerativeAI(**default_params)


def execute_tool_calls(ai_msg: AIMessage, tools: List, messages: List) -> List:
    """Execute tool calls from an AI message and return updated messages.

    Args:
        ai_msg: AI message potentially containing tool calls
        tools: List of available tools
        messages: Current message list

    Returns:
        Updated message list with tool results
    """
    if not ai_msg.tool_calls:
        return messages

    for tool_call in ai_msg.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        # Find and execute the tool
        tool_found = False
        for tool in tools:
            if tool.name == tool_name:
                tool_found = True
                try:
                    result = tool.invoke(tool_args)
                    messages.append(
                        ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                    )
                except Exception as e:
                    logging.error(f"Error executing tool {tool_name}: {e}")
                    messages.append(
                        ToolMessage(
                            content=f"Error: {str(e)}", tool_call_id=tool_call["id"]
                        )
                    )
                break

        if not tool_found:
            logging.warning(f"Tool {tool_name} not found")
            messages.append(
                ToolMessage(
                    content=f"Error: Tool {tool_name} not found",
                    tool_call_id=tool_call["id"],
                )
            )

    return messages
