"""
Retail Analytics React Agent using LangGraph

This module creates a React agent that can execute retail analytics queries,
format results, and provide conversational analytics capabilities using LangGraph.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, TypedDict, Union

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from loguru import logger
from pydantic import BaseModel, Field

# Import your existing components
from src.query_tools.bigquery_connector import BigQuerySQLExecutor
from src.query_tools.database import QueryConfig, QueryResult
from src.query_tools.query_formatter import (
    LLMQueryFormatter,
    format_query_for_llm_analysis,
    format_query_params,
    format_query_result,
)
# from src.retail_analytics_executor import (
#     RetailAnalyticsExecutor,
#     create_retail_analytics_executor,
# )


# Agent State
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage, SystemMessage, ToolMessage]]
    query_results: Optional[Dict[str, Any]]
    analysis_context: Optional[str]


# Tool Input Models
class WeeklyStorePerformanceInput(BaseModel):
    """Input for weekly store performance metrics query."""

    salesorg: str = Field(description="Sales organization identifier (e.g., '1005')")
    regions: List[str] = Field(
        description="List of regions to analyze (e.g., ['NSW', 'VIC', 'QLD'])"
    )
    timezone_1: str = Field(
        default="Australia/Sydney", description="Timezone for first comparison period"
    )
    timezone_2: str = Field(
        default="Australia/Sydney", description="Timezone for second comparison period"
    )
    timezone_3: str = Field(
        default="Australia/Sydney", description="Timezone for third comparison period"
    )
    timezone_4: str = Field(
        default="Australia/Sydney", description="Timezone for fourth comparison period"
    )
    comparison_days_1: int = Field(default=92, description="Days back for first period")
    comparison_days_2: int = Field(default=1, description="Days back for second period")
    comparison_days_3: int = Field(
        default=456, description="Days back for third period"
    )
    comparison_days_4: int = Field(
        default=364, description="Days back for fourth period"
    )
    comparison_days_offset_1: int = Field(
        default=-1, description="Offset for first period"
    )
    comparison_days_offset_2: int = Field(
        default=-1, description="Offset for second period"
    )


class SalesBudgetInput(BaseModel):
    """Input for sales vs budget comparison query."""

    salesorg: str = Field(description="Sales organization identifier")
    comparison_timehierarchy: str = Field(
        description="Time hierarchy for comparison (e.g., 'FY2024Q4')"
    )


class SubcategoryAnalysisInput(BaseModel):
    """Input for subcategory sales analysis query."""

    fiscal_years: List[str] = Field(description="List of fiscal years to analyze")
    salesorg: str = Field(default="1005", description="Sales organization identifier")
    merchandise_managers: Optional[List[str]] = Field(
        default=["MM924_AU", "MM920_AU"],
        description="List of merchandise manager codes",
    )


class StoreTrendInput(BaseModel):
    """Input for store sales trend analysis query."""

    salesorg: str = Field(description="Sales organization identifier")
    fiscal_year: str = Field(description="Fiscal year for analysis")


class QueryAnalysisInput(BaseModel):
    """Input for query analysis."""

    query: str = Field(description="SQL query to analyze")
    params: Dict[str, Any] = Field(description="Query parameters")
    analysis_type: str = Field(
        default="explain", description="Type of analysis: explain, optimize, validate"
    )


# Retail Analytics Tools
class RetailAnalyticsTools:
    """Collection of retail analytics tools for the React agent."""

    def __init__(
        self, project_id: Optional[str] = None, credentials_path: Optional[str] = None
    ):
        """Initialize retail analytics tools."""
        self.executor = BigQuerySQLExecutor(project_id=project_id)
        self.llm_formatter = LLMQueryFormatter()
        logger.info("Retail Analytics Tools initialized")

    @tool("weekly_store_performance", args_schema=WeeklyStorePerformanceInput)
    def weekly_store_performance(self, **kwargs) -> str:
        """
        Execute weekly store performance metrics query.

        Shows weekly sales, stock adjustments, stock loss, and customer counts by store.
        Compares This Year (TY) vs Last Year (LY) performance with competitor distance analysis.

        Returns formatted results with key metrics and insights.
        """
        try:
            logger.info(
                f"Executing weekly store performance query for salesorg: {kwargs.get('salesorg')}"
            )

            result = self.executor.weekly_store_performance_metrics(**kwargs)

            # Format result for LLM consumption
            formatted_result = format_query_result(result, "summary")

            # Add business insights
            insights = []
            if result.row_count > 0:
                insights.append(
                    f"✅ Successfully retrieved {result.row_count} store performance records"
                )
                insights.append(
                    f"⏱️ Query executed in {result.execution_time_seconds:.2f} seconds"
                )

                if result.metadata.get("bytes_processed"):
                    insights.append(
                        f"📊 Processed {result.metadata['bytes_processed']:,} bytes"
                    )

                # Sample data insights
                if result.rows:
                    sample_row = result.rows[0]
                    if isinstance(sample_row, dict):
                        if "sales" in sample_row:
                            insights.append(
                                f"💰 Sample store sales: ${sample_row['sales']:,.2f}"
                            )
                        if "customers" in sample_row:
                            insights.append(
                                f"👥 Sample customer count: {sample_row['customers']:,}"
                            )

            return f"{formatted_result}\n\nKey Insights:\n" + "\n".join(insights)

        except Exception as e:
            logger.error(f"Weekly store performance query failed: {e}")
            return f"❌ Error executing weekly store performance query: {str(e)}"

    @tool("sales_vs_budget", args_schema=SalesBudgetInput)
    def sales_vs_budget(self, salesorg: str, comparison_timehierarchy: str) -> str:
        """
        Execute sales vs budget comparison query.

        Compares actual sales against operational and merchandise budgets for a specific time period.
        Provides variance analysis and performance indicators.

        Args:
            salesorg: Sales organization identifier
            comparison_timehierarchy: Time hierarchy for comparison period

        Returns formatted budget comparison results.
        """
        try:
            logger.info(
                f"Executing sales vs budget query for {salesorg}, period: {comparison_timehierarchy}"
            )

            result = self.executor.sales_vs_budget_by_fiscal_period(
                salesorg, comparison_timehierarchy
            )

            formatted_result = format_query_result(result, "summary")

            # Add budget analysis insights
            insights = []
            if result.row_count > 0:
                insights.append(
                    f"📈 Budget comparison data for {result.row_count} fiscal periods"
                )

                # Calculate budget variance if data available
                if result.rows:
                    total_actual = sum(
                        row.get("actuals_sales", 0)
                        for row in result.rows
                        if isinstance(row, dict)
                    )
                    total_ops_budget = sum(
                        row.get("ops_budget", 0)
                        for row in result.rows
                        if isinstance(row, dict)
                    )

                    if total_ops_budget > 0:
                        variance_pct = (
                            (total_actual - total_ops_budget) / total_ops_budget
                        ) * 100
                        status = "over" if variance_pct > 0 else "under"
                        insights.append(
                            f"🎯 Performance: {abs(variance_pct):.1f}% {status} operations budget"
                        )

            return f"{formatted_result}\n\nBudget Analysis:\n" + "\n".join(insights)

        except Exception as e:
            logger.error(f"Sales vs budget query failed: {e}")
            return f"❌ Error executing sales vs budget query: {str(e)}"

    @tool("subcategory_analysis", args_schema=SubcategoryAnalysisInput)
    def subcategory_analysis(
        self,
        fiscal_years: List[str],
        salesorg: str = "1005",
        merchandise_managers: Optional[List[str]] = None,
    ) -> str:
        """
        Execute weekly sales and gross profit analysis by subcategory.

        Analyzes subcategory performance including sales, items sold, scanback, and gross profit.
        Provides insights into merchandise department performance trends.

        Returns detailed subcategory performance analysis.
        """
        try:
            logger.info(f"Executing subcategory analysis for years: {fiscal_years}")

            result = self.executor.weekly_sales_and_gross_profit_by_subcategory(
                fiscal_years, salesorg, merchandise_managers
            )

            formatted_result = format_query_result(result, "summary")

            # Add subcategory insights
            insights = []
            if result.row_count > 0:
                insights.append(
                    f"🏪 Subcategory data for {result.row_count} records across {len(fiscal_years)} years"
                )

                # Calculate top performers if data available
                if result.rows:
                    # Group by subcategory and calculate totals
                    subcategory_totals = {}
                    for row in result.rows:
                        if (
                            isinstance(row, dict)
                            and "subcategory" in row
                            and "sales" in row
                        ):
                            subcat = row["subcategory"]
                            sales = float(row.get("sales", 0))
                            subcategory_totals[subcat] = (
                                subcategory_totals.get(subcat, 0) + sales
                            )

                    if subcategory_totals:
                        top_subcat = max(subcategory_totals.items(), key=lambda x: x[1])
                        insights.append(
                            f"🥇 Top subcategory: {top_subcat[0]} (${top_subcat[1]:,.2f})"
                        )

            return f"{formatted_result}\n\nSubcategory Insights:\n" + "\n".join(
                insights
            )

        except Exception as e:
            logger.error(f"Subcategory analysis query failed: {e}")
            return f"❌ Error executing subcategory analysis: {str(e)}"

    @tool("store_trends", args_schema=StoreTrendInput)
    def store_trends(self, salesorg: str, fiscal_year: str) -> str:
        """
        Execute store sales profitability weekly trend analysis.

        Analyzes weekly and monthly trends in store sales, units, transactions, and gross profit.
        Includes competitor distance analysis and sales channel breakdown.

        Returns comprehensive store trend analysis.
        """
        try:
            logger.info(
                f"Executing store trends analysis for {salesorg}, FY{fiscal_year}"
            )

            result = self.executor.store_sales_profitability_weekly_trend(
                salesorg, fiscal_year
            )

            formatted_result = format_query_result(result, "summary")

            # Add trend insights
            insights = []
            if result.row_count > 0:
                insights.append(
                    f"📊 Store trend data: {result.row_count} records for FY{fiscal_year}"
                )

                # Calculate trend metrics if data available
                if result.rows:
                    total_sales = sum(
                        float(row.get("sales", 0))
                        for row in result.rows
                        if isinstance(row, dict)
                    )
                    total_transactions = sum(
                        float(row.get("transactions", 0))
                        for row in result.rows
                        if isinstance(row, dict)
                    )

                    if total_transactions > 0:
                        avg_basket = total_sales / total_transactions
                        insights.append(f"🛒 Average basket size: ${avg_basket:.2f}")

                    insights.append(f"💰 Total sales analyzed: ${total_sales:,.2f}")

            return f"{formatted_result}\n\nTrend Analysis:\n" + "\n".join(insights)

        except Exception as e:
            logger.error(f"Store trends query failed: {e}")
            return f"❌ Error executing store trends analysis: {str(e)}"

    @tool("analyze_query", args_schema=QueryAnalysisInput)
    def analyze_query(
        self, query: str, params: Dict[str, Any], analysis_type: str = "explain"
    ) -> str:
        """
        Analyze a SQL query for explanation, optimization, or validation.

        Provides detailed analysis of query structure, business logic, and recommendations.

        Args:
            query: SQL query to analyze
            params: Query parameters
            analysis_type: Type of analysis (explain, optimize, validate)

        Returns formatted query analysis.
        """
        try:
            logger.info(f"Analyzing query with type: {analysis_type}")

            # Format query for LLM analysis
            analysis_prompt = format_query_for_llm_analysis(
                query, params, analysis_type=analysis_type
            )

            # Add structured analysis
            insights = []
            insights.append(f"🔍 Query Analysis Type: {analysis_type.title()}")
            insights.append(f"📝 Parameters: {len(params)} parameters detected")

            # Basic query characteristics
            query_upper = query.upper()
            if "SELECT" in query_upper:
                insights.append("📊 Query Type: SELECT (Data Retrieval)")
            if "GROUP BY" in query_upper:
                insights.append("📈 Aggregation: Contains GROUP BY clause")
            if "CASE" in query_upper:
                insights.append("🔀 Logic: Contains conditional CASE statements")
            if "@" in query:
                insights.append("🔒 Security: Uses parameterized queries (Good!)")

            return f"{analysis_prompt}\n\nQuick Analysis:\n" + "\n".join(insights)

        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return f"❌ Error analyzing query: {str(e)}"

    @tool("test_connection")
    def test_connection(self) -> str:
        """Test BigQuery connection and verify system health.

        Returns connection status and system information.
        """
        try:
            logger.info("Testing BigQuery connection")

            if self.executor.test_connection():
                return "✅ BigQuery connection successful! System is ready for analytics queries."
            else:
                return "❌ BigQuery connection failed. Please check credentials and network connectivity."

        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return f"❌ Connection test error: {str(e)}"


# React Agent
class RetailAnalyticsAgent:
    """LangGraph React Agent for Retail Analytics."""

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        project_id: Optional[str] = None,
        credentials_path: Optional[str] = None,
        temperature: float = 0.1,
    ):
        """
        Initialize the Retail Analytics Agent.

        Args:
            model_name: LLM model to use
            project_id: GCP project ID for BigQuery
            credentials_path: Path to GCP credentials
            temperature: LLM temperature for responses
        """
        self.tools_instance = RetailAnalyticsTools(project_id, credentials_path)

        # Create tools list
        self.tools = [
            self.tools_instance.weekly_store_performance,
            self.tools_instance.sales_vs_budget,
            self.tools_instance.subcategory_analysis,
            self.tools_instance.store_trends,
            self.tools_instance.analyze_query,
            self.tools_instance.test_connection,
        ]

        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)

        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Create graph
        self.graph = self._create_graph()

        logger.info(f"Retail Analytics Agent initialized with {len(self.tools)} tools")

    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow."""

        def should_continue(state: AgentState):
            """Determine if we should continue or end."""
            messages = state["messages"]
            last_message = messages[-1]

            # If the last message has tool calls, continue to tools
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            else:
                return END

        def call_model(state: AgentState):
            """Call the LLM with current state."""
            messages = state["messages"]

            # Add system message if this is the first call
            if not any(isinstance(msg, SystemMessage) for msg in messages):
                system_message = SystemMessage(content=self._get_system_prompt())
                messages = [system_message] + messages

            response = self.llm_with_tools.invoke(messages)
            return {"messages": messages + [response]}

        # Create workflow
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(self.tools))

        # Set entry point
        workflow.set_entry_point("agent")

        # Add edges
        workflow.add_conditional_edges("agent", should_continue)
        workflow.add_edge("tools", "agent")

        # Add memory
        memory = MemorySaver()

        return workflow.compile(checkpointer=memory)

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent."""
        return """You are a Retail Analytics Assistant specializing in Woolworths Group business intelligence and data analysis.

Your capabilities include:
🏪 **Store Performance Analysis**: Weekly metrics, TY vs LY comparisons, competitor analysis
📊 **Financial Analysis**: Sales vs budget comparisons, variance analysis, performance tracking  
🛍️ **Category Management**: Subcategory performance, merchandise analysis, trend identification
📈 **Trend Analysis**: Weekly/monthly patterns, seasonality, growth analysis
🔍 **Query Analysis**: SQL explanation, optimization recommendations, validation

**Guidelines:**
- Always provide actionable business insights, not just raw data
- Use clear, professional language with relevant emojis for readability
- Explain complex concepts in business terms
- Highlight key performance indicators and trends
- Suggest follow-up analyses when appropriate
- Format numbers with proper currency and comma separators

**Data Context:**
- Sales Organization 1005 = Woolworths Supermarkets  
- Regions: NSW, VIC, QLD, WA, SA, NT, TAS
- Fiscal years follow Woolworths calendar
- TY = This Year, LY = Last Year
- Competitor analysis includes Coles proximity data

Start by testing the connection if this is a new session, then assist with any retail analytics questions or requests."""

    async def arun(self, message: str, thread_id: str = "default") -> str:
        """
        Run the agent asynchronously.

        Args:
            message: User message/query
            thread_id: Thread ID for conversation memory

        Returns:
            Agent response
        """
        try:
            # Create initial state
            initial_state = {
                "messages": [HumanMessage(content=message)],
                "query_results": None,
                "analysis_context": None,
            }

            # Configure with thread
            config = {"configurable": {"thread_id": thread_id}}

            # Run the graph
            final_state = await self.graph.ainvoke(initial_state, config)

            # Get the last AI message
            last_message = final_state["messages"][-1]
            if isinstance(last_message, AIMessage):
                return last_message.content
            else:
                return "I apologize, but I couldn't process your request properly. Please try again."

        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return f"❌ I encountered an error: {str(e)}. Please try rephrasing your request."

    def run(self, message: str, thread_id: str = "default") -> str:
        """Run the agent synchronously.

        Args:
            message: User message/query
            thread_id: Thread ID for conversation memory

        Returns:
            Agent response
        """
        return asyncio.run(self.arun(message, thread_id))

    def stream(self, message: str, thread_id: str = "default"):
        """Stream agent responses.

        Args:
            message: User message/query
            thread_id: Thread ID for conversation memory

        Yields:
            Agent response chunks
        """
        print("streaming")
        try:
            initial_state = {
                "messages": [HumanMessage(content=message)],
                "query_results": None,
                "analysis_context": None,
            }

            config = {"configurable": {"thread_id": thread_id}}

            for chunk in self.graph.stream(initial_state, config):
                print("testing")
                if "agent" in chunk:
                    if chunk["agent"]["messages"]:
                        last_message = chunk["agent"]["messages"][-1]
                        if isinstance(last_message, AIMessage) and last_message.content:
                            yield last_message.content

        except Exception as e:
            logger.error(f"Agent streaming failed: {e}")
            yield f"❌ Streaming error: {str(e)}"


# Convenience function
def create_retail_analytics_agent(
    model_name: str = "gemini-2.0-flash",
    project_id: Optional[str] = None,
    credentials_path: Optional[str] = None,
    temperature: float = 0.1,
) -> RetailAnalyticsAgent:
    """
    Create a configured Retail Analytics Agent.

    Args:
        model_name: LLM model to use
        project_id: GCP project ID for BigQuery
        credentials_path: Path to GCP credentials
        temperature: LLM temperature

    Returns:
        Configured RetailAnalyticsAgent
    """
    return RetailAnalyticsAgent(model_name, project_id, credentials_path, temperature)


# Example usage
if __name__ == "__main__":
    import os

    # Configure logging
    logger.configure(
        handlers=[
            {"sink": "logs/agent.log", "level": "DEBUG", "rotation": "10 MB"},
            {
                "sink": lambda msg: print(msg),
                "level": "INFO",
                "format": "{time} | {level} | {message}",
            },
        ]
    )

    project_id = os.getenv("GCP_PROJECT_ID") or "wiq-gen-ai-rd-dev"
    credentials_path = os.getenv("GCP_CREDENTIALS_PATH")
    print("project_id", project_id)
    print("credentials_path", credentials_path)

    # Create agent
    agent = create_retail_analytics_agent(
        project_id=project_id,
        credentials_path=credentials_path,
    )

    # Example interactions
    test_queries = [
        "Test the BigQuery connection",
        "Show me weekly store performance for Woolworths supermarkets in NSW and VIC",
        "Compare sales vs budget for Q4 2024",
        "Analyze subcategory performance for fiscal years 2023 and 2024",
        "What are the sales trends for stores in fiscal year 2024?",
    ]

    print("🤖 Retail Analytics Agent Ready!")
    print("=" * 50)

    for query in test_queries:
        print(f"\n👤 User: {query}")
        print("huh")
        print("🤖 Agent:", end=" ")

        # Stream response
        for chunk in agent.stream(query):
            print(chunk, end="", flush=True)
        print("\n" + "-" * 50)
