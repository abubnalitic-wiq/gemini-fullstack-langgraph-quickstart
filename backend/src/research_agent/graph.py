"""LangGraph-based research agent for automated web research and summarization."""

import logging
import os
from typing import Any

from dotenv import load_dotenv
from google.genai import Client
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from src.query_tools.financial_data_tools import financial_analysis_tools
from src.research_agent.configuration import Configuration
from src.research_agent.history_summariser import (
    add_to_state,
    create_history_context,
    get_message_tokens,
    summarize_conversation,
    truncate_messages,
)
from src.research_agent.llm_utils import (
    create_llm,
    execute_tool_calls,
    get_latest_user_message,
)
from src.research_agent.prompts import (
    CHAT_PROMPT,
    FINANCIAL_ANALYST_PROMPT,
    FINANCIAL_QA_PROMPT,
    ROUTING_PROMPT,
    answer_instructions,
    get_current_date,
    query_writer_instructions,
    reflection_instructions,
    web_searcher_instructions,
)
from src.research_agent.research_utils import (
    get_citations,
    get_research_topic,
    insert_citation_markers,
    resolve_urls,
)
from src.research_agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from src.research_agent.tools_and_schemas import Reflection, SearchQueryList

load_dotenv()
logging.basicConfig(level=logging.DEBUG)

# Validate API key at startup
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY is not set")

# Initialize Google GenAI client once
genai_client = Client(api_key=os.getenv("GEMINI_API_KEY"))


# ===== Routing =====


class RouteDecision(BaseModel):
    """Routing decision for user query"""

    route: str = Field(
        description="One of: 'web_research', 'detailed_financial_report', 'general_financial_question', 'general_discussion'"
    )
    confidence: float = Field(description="Confidence in the routing decision (0-1)")
    reasoning: str = Field(description="Brief explanation for the routing decision")


def route_query(state: OverallState, config: RunnableConfig) -> dict[str, Any]:
    """Determines which of 4 routes to take based on user query."""
    state = add_to_state(state)
    latest_message = get_latest_user_message(state["messages"])

    if not latest_message:
        logging.warning("No user message found, defaulting to general_discussion")
        return {"query_type": "general_discussion"}

    llm = create_llm(config, "router_model", temperature=0.3)
    history_context = (
        create_history_context(
            state["conversation_history"], llm, max_history_tokens=1000
        )
        or "No previous conversation"
    )
    structured_llm = llm.with_structured_output(RouteDecision)

    try:
        decision = structured_llm.invoke(
            ROUTING_PROMPT.format(query=latest_message, history_context=history_context)
        )
        query_type = decision.route

        logging.info(f"Query: '{latest_message[:100]}...'")
        logging.info(f"Routed to: {query_type} (confidence: {decision.confidence})")
        logging.info(f"Reasoning: {decision.reasoning}")

        # Validate route name
        valid_routes = [
            "web_research",
            "detailed_financial_report",
            "general_financial_question",
            "general_discussion",
        ]
        if query_type not in valid_routes:
            logging.warning(
                f"Invalid route '{query_type}', defaulting to general_discussion"
            )
            query_type = "general_discussion"

    except Exception as e:
        logging.error(
            f"Error in structured LLM routing: {e}, falling back to general_discussion"
        )
        query_type = "general_discussion"

    return {"query_type": query_type, "state": state}


# ===== Web Research Nodes =====


def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """Generate search queries based on the user's question."""
    # state = add_to_state(state)
    configurable = Configuration.from_runnable_config(config)

    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    logging.info("[generate_query] Generating search queries based on user question.")

    llm = create_llm(config, "query_generator_model", temperature=1.0)

    history_context = (
        create_history_context(
            state["conversation_history"], llm, max_history_tokens=1500
        )
        or "No previous financial discussions."
    )

    structured_llm = llm.with_structured_output(SearchQueryList)

    formatted_prompt = query_writer_instructions.format(
        current_date=get_current_date(),
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
        history_context=history_context,
    )

    result = structured_llm.invoke(formatted_prompt)
    return {"query_list": result.query}


def continue_to_web_research(state: QueryGenerationState):
    """Send search queries to web research nodes."""
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["query_list"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """Perform web research using Google Search API."""
    configurable = Configuration.from_runnable_config(config)

    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
    )

    logging.info(f"[web_research] formatted_prompt:\n{formatted_prompt}")

    # Use native Google API for grounding metadata
    response = genai_client.models.generate_content(
        model=configurable.query_generator_model,
        contents=formatted_prompt,
        config={
            "tools": [{"google_search": {}}],
            "temperature": 0,
        },
    )

    logging.info(f"[response] response:\n{response}")

    # Process response
    resolved_urls = resolve_urls(
        response.candidates[0].grounding_metadata.grounding_chunks, state["id"]
    )
    citations = get_citations(response, resolved_urls)
    modified_text = insert_citation_markers(response.text, citations)
    sources_gathered = [item for citation in citations for item in citation["segments"]]

    return {
        "sources_gathered": sources_gathered,
        "search_query": [state["search_query"]],
        "web_research_result": [modified_text],
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """Identify knowledge gaps and generate follow-up queries."""
    configurable = Configuration.from_runnable_config(config)

    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model") or configurable.reasoning_model

    formatted_prompt = reflection_instructions.format(
        current_date=get_current_date(),
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )

    llm = create_llm(config, model_type="reasoning_model", temperature=1.0)
    llm._model = reasoning_model  # Override with specific reasoning model

    result = llm.with_structured_output(Reflection).invoke(formatted_prompt)

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(
    state: ReflectionState, config: RunnableConfig
) -> str | list[Send]:
    """Determine next step in research flow."""
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )

    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


def finalize_answer(state: OverallState, config: RunnableConfig) -> dict[str, Any]:
    """Finalize the research summary."""

    formatted_prompt = answer_instructions.format(
        current_date=get_current_date(),
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    llm = create_llm(config, model_type="reasoning_model", temperature=0)

    result = llm.invoke(formatted_prompt)

    # Replace short URLs with original URLs
    unique_sources = []
    for source in state["sources_gathered"]:
        if source["short_url"] in result.content:
            result.content = result.content.replace(
                source["short_url"], source["value"]
            )
            unique_sources.append(source)

    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": unique_sources,
    }


# ===== Financial Nodes =====


def financial_qa(state: OverallState, config: RunnableConfig) -> dict[str, Any]:
    """Handle quick financial questions without generating full reports."""
    user_question = get_latest_user_message(state["messages"])

    if not user_question:
        return {
            "messages": [AIMessage(content="I couldn't find a question to answer.")]
        }

    llm = create_llm(config, "analyst_model", temperature=0.2)

    history_context = (
        create_history_context(
            state["conversation_history"], llm, max_history_tokens=1500
        )
        or "No previous financial discussions."
    )

    formatted_prompt = FINANCIAL_QA_PROMPT.format(
        user_question=user_question, history_context=history_context
    )

    llm_with_tools = llm.bind_tools(financial_analysis_tools)

    messages = [HumanMessage(content=formatted_prompt)]
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)

    # Execute tool calls if any
    messages = execute_tool_calls(ai_msg, financial_analysis_tools, messages)

    # Get final response if tools were called
    if ai_msg.tool_calls:
        final_response = llm.invoke(messages)
        return {"messages": [final_response]}

    return {"messages": [ai_msg]}


def general_chat(state: OverallState, config: RunnableConfig) -> dict[str, Any]:
    """Handle general chat with conversation history."""
    # state = add_to_state(state)
    user_question = get_latest_user_message(state["messages"])

    if not user_question:
        return {"messages": [AIMessage(content="Hello! How can I help you today?")]}

    llm = create_llm(config, "general_chat_model", temperature=0.7)

    # Create history context
    history_context = create_history_context(
        state["conversation_history"], llm, max_history_tokens=1500
    )

    # Create messages with history
    messages = [SystemMessage(content=CHAT_PROMPT)]

    if history_context:
        messages.append(
            SystemMessage(content=f"Previous conversation:\n{history_context}")
        )

    messages.append(HumanMessage(content=user_question))

    response = llm.invoke(messages)

    return {"messages": [response]}


def detailed_financial_report(
    state: OverallState, config: RunnableConfig
) -> dict[str, Any]:
    """Generate comprehensive financial reports using financial tools."""
    user_query = get_latest_user_message(state["messages"])

    if not user_query:
        return {
            "messages": [
                AIMessage(content="Please provide a query for the financial report.")
            ],
            "sources_gathered": [],
        }

    llm = create_llm(config, "query_generator_model", temperature=0)

    history_context = (
        create_history_context(
            state["conversation_history"], llm, max_history_tokens=1500
        )
        or "No previous financial discussions."
    )
    financial_prompt = FINANCIAL_ANALYST_PROMPT.format(
        user_query=user_query, history_context=history_context
    )

    llm_with_tools = llm.bind_tools(financial_analysis_tools)

    logging.info(f"Generated financial prompt: {financial_prompt}")

    messages = [HumanMessage(content=financial_prompt)]
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)

    # Execute tool calls
    messages = execute_tool_calls(ai_msg, financial_analysis_tools, messages)

    # Get final response if tools were called
    if ai_msg.tool_calls:
        final_response = llm.invoke(messages)
        return {
            "messages": [final_response],
            "sources_gathered": [],
        }

    return {
        "messages": [ai_msg],
        "sources_gathered": [],
    }


# ===== Truncation Node =====


def truncate_history(state: OverallState, config: RunnableConfig) -> dict[str, Any]:
    """Dedicated node to truncate conversation history at the end of each flow."""
    configurable = Configuration.from_runnable_config(config)

    # Initialize conversation_history if not present
    if "conversation_history" not in state:
        state["conversation_history"] = []

    # Get current conversation history
    current_history = state["conversation_history"].copy()

    # Add the latest exchange to history
    latest_user_msg = None
    latest_ai_msg = None

    # Find the latest user message and AI response
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) and not latest_user_msg:
            latest_user_msg = msg
        elif isinstance(msg, AIMessage) and not latest_ai_msg:
            latest_ai_msg = msg

        if latest_user_msg and latest_ai_msg:
            break

    # Add new messages to history
    if latest_user_msg and latest_user_msg not in current_history:
        current_history.append(latest_user_msg)
    if latest_ai_msg and latest_ai_msg not in current_history:
        current_history.append(latest_ai_msg)

    # Get truncation parameters from config
    max_history_tokens = getattr(configurable, "max_history_tokens", 4000)
    keep_first = getattr(configurable, "keep_first_messages", 2)
    keep_last = getattr(configurable, "keep_last_messages", 6)
    summarize_threshold = getattr(configurable, "summarize_threshold", 6000)

    # Check if summarization is needed
    total_tokens = sum(get_message_tokens(msg) for msg in current_history)

    if total_tokens > summarize_threshold:
        # Create LLM for summarization
        llm = create_llm(config, "query_generator_model", temperature=0.3)

        # Find messages to summarize (older ones)
        summarize_count = len(current_history) // 3  # Summarize oldest third
        messages_to_summarize = current_history[:summarize_count]
        remaining_messages = current_history[summarize_count:]

        try:
            summary = summarize_conversation(
                messages_to_summarize, llm, max_summary_tokens=500
            )
            # Create a system message with the summary
            summary_msg = SystemMessage(
                content=f"Previous Conversation Summary:\n{summary}"
            )
            current_history = [summary_msg] + remaining_messages
            logging.info("Conversation history summarized")
        except Exception as e:
            logging.error(f"Failed to summarize conversation: {e}")

    # Truncate if still too long
    truncated_history, was_truncated = truncate_messages(
        current_history,
        max_tokens=max_history_tokens,
        keep_first=keep_first,
        keep_last=keep_last,
    )

    if was_truncated:
        logging.info(
            f"Conversation history truncated from {len(current_history)} to {len(truncated_history)} messages"
        )

    return {"conversation_history": truncated_history}


# ===== Graph Construction =====


def build_graph():
    """Build and compile the agent graph."""
    builder = StateGraph(OverallState, config_schema=Configuration)

    # Add nodes
    builder.add_node("route_query", route_query)
    builder.add_node("generate_query", generate_query)
    builder.add_node("web_research", web_research)
    builder.add_node("reflection", reflection)
    builder.add_node("finalize_answer", finalize_answer)
    builder.add_node("detailed_financial_report", detailed_financial_report)
    builder.add_node("financial_qa", financial_qa)
    builder.add_node("general_chat", general_chat)
    builder.add_node("truncate_history", truncate_history)

    # Set entry point
    builder.set_entry_point("route_query")

    # Add routing edges
    builder.add_conditional_edges(
        "route_query",
        lambda state: state.get("query_type", "general_discussion"),
        {
            "web_research": "generate_query",
            "detailed_financial_report": "detailed_financial_report",
            "general_financial_question": "financial_qa",
            "general_discussion": "general_chat",
        },
    )

    # Web research flow
    builder.add_conditional_edges(
        "generate_query", continue_to_web_research, ["web_research"]
    )
    builder.add_edge("web_research", "reflection")
    builder.add_conditional_edges(
        "reflection",
        evaluate_research,
        ["web_research", "finalize_answer"],
    )

    # Terminal edges
    builder.add_edge("finalize_answer", "truncate_history")
    builder.add_edge("detailed_financial_report", "truncate_history")
    builder.add_edge("financial_qa", "truncate_history")
    builder.add_edge("general_chat", "truncate_history")

    builder.add_edge("truncate_history", END)

    return builder.compile(name="pro-search-agent")


# Create the graph
graph = build_graph()
