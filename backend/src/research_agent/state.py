from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import TypedDict
from typing_extensions import Annotated

from langgraph.graph import add_messages


class OverallState(TypedDict):
    """Represents the overall state of the research agent.

    Attributes:
        messages: List of messages exchanged in the session.
        search_query: List of search queries generated.
        web_research_result: List of results from web research.
        sources_gathered: List of sources collected during research.
        initial_search_query_count: Number of initial search queries.
        max_research_loops: Maximum allowed research loops.
        research_loop_count: Current count of research loops performed.
        reasoning_model: Name of the reasoning model in use.
    """

    messages: Annotated[list[str], add_messages]
    search_query: Annotated[list[str], operator.add]
    web_research_result: Annotated[list[str], operator.add]
    sources_gathered: Annotated[list[str], operator.add]
    initial_search_query_count: int
    max_research_loops: int
    research_loop_count: int
    reasoning_model: str
    query_type: str = ""


class ReflectionState(TypedDict):
    """Represents the state after reflecting on research results.

    Attributes:
        is_sufficient: Whether the gathered information is sufficient.
        knowledge_gap: Description of any remaining knowledge gap.
        follow_up_queries: List of follow-up queries to address gaps.
        research_loop_count: Current count of research loops performed.
        number_of_ran_queries: Number of queries executed so far.
    """

    is_sufficient: bool
    knowledge_gap: str
    follow_up_queries: Annotated[list[str], operator.add]
    research_loop_count: int
    number_of_ran_queries: int


class Query(TypedDict):
    """Represents a single search query and its rationale.

    Attributes:
        query: The search query string.
        rationale: The reasoning behind the query.
    """

    query: str
    rationale: str


class QueryGenerationState(TypedDict):
    """Represents the state during query generation.

    Attributes:
        query_list: List of queries to be executed.
    """

    query_list: list[Query]


class WebSearchState(TypedDict):
    """Represents the state for a single web search operation.

    Attributes:
        search_query: The search query string.
        id: Unique identifier for the search operation.
    """

    search_query: str
    id: str


@dataclass(kw_only=True)
class SearchStateOutput:
    """Output state containing the running summary or final report.

    Attributes:
        running_summary: The final report or summary of the research.
    """

    running_summary: str = field(default="")  # Final report
