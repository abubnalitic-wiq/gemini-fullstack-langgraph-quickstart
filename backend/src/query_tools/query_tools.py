from src.query_tools.query_runner import run_bigquery_query
from src.query_tools import query_functions

# Wrap each query function with the BigQuery runner
weekly_store_performance_metrics_tool = run_bigquery_query(
    query_functions.weekly_store_performance_metrics
)
sales_vs_budget_by_fiscal_period_tool = run_bigquery_query(
    query_functions.sales_vs_budget_by_fiscal_period
)
weekly_sales_and_gross_profit_by_subcategory_tool = run_bigquery_query(
    query_functions.weekly_sales_and_gross_profit_by_subcategory
)
store_sales_profitability_weekly_trend_tool = run_bigquery_query(
    query_functions.store_sales_profitability_weekly_trend
)

# Optionally, create a registry for LangGraph
TOOLS = [
    {
        "name": "weekly_store_performance_metrics",
        "description": "Get weekly store performance metrics for given salesorg and regions.",
        "function": weekly_store_performance_metrics_tool,
    },
    {
        "name": "sales_vs_budget_by_fiscal_period",
        "description": "Compare sales vs budget by fiscal period for a salesorg.",
        "function": sales_vs_budget_by_fiscal_period_tool,
    },
    {
        "name": "weekly_sales_and_gross_profit_by_subcategory",
        "description": "Get weekly sales and gross profit by subcategory for given fiscal years and managers.",
        "function": weekly_sales_and_gross_profit_by_subcategory_tool,
    },
    {
        "name": "store_sales_profitability_weekly_trend",
        "description": "Get store sales and profitability weekly trends for a salesorg and fiscal year.",
        "function": store_sales_profitability_weekly_trend_tool,
    },
]
