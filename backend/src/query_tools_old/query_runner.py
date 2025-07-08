from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple

from src.query_tools.bigquery_connector import BigQuerySQLExecutor
from src.query_tools.database import QueryConfig

executor = BigQuerySQLExecutor(project_id="wiq-gen-ai-rd-dev")


def run_bigquery_query(query_func: Callable[..., Tuple[str, Dict]]) -> Callable:
    """Wrap a query generation function that returns (query, params).

    Executes the query using BigQuerySQLExecutor and returns the results.
    """

    @wraps(query_func)
    def wrapper(*args, config: Optional[QueryConfig] = None, **kwargs) -> Any:
        query, params = query_func(*args, **kwargs)
        result = executor.execute_query(query, config=config, parameters=params)
        return result

    return wrapper
