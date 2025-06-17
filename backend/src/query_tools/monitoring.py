"""Decorator for monitoring query execution"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from src.query_tools.database import SQLQueryExecutor, QueryResult, QueryConfig


class QueryMetrics:
    """Simple metrics collection"""

    def __init__(self):
        self.queries: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)

    def record_query(self, query: str, result: QueryResult, config: QueryConfig):
        """Record query execution metrics"""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "query": query[:200] + "..."
            if len(query) > 200
            else query,  # Truncate long queries
            "execution_time": result.execution_time_seconds,
            "row_count": result.row_count,
            "is_truncated": result.is_truncated,
            "bytes_processed": result.metadata.get("bytes_processed", 0),
            "cache_hit": result.metadata.get("cache_hit", False),
        }

        self.queries.append(metric)

        # Log expensive queries
        if result.execution_time_seconds > 5.0:
            self.logger.warning(
                f"Slow query detected: {result.execution_time_seconds:.2f}s"
            )

        bytes_processed = result.metadata.get("bytes_processed", 0)
        if bytes_processed > 100 * 1024 * 1024:  # 100MB
            self.logger.warning(
                f"Large query detected: {bytes_processed / (1024*1024):.1f}MB processed"
            )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        if not self.queries:
            return {"total_queries": 0}

        execution_times = [q["execution_time"] for q in self.queries]
        bytes_processed = [q["bytes_processed"] for q in self.queries]

        return {
            "total_queries": len(self.queries),
            "avg_execution_time": sum(execution_times) / len(execution_times),
            "max_execution_time": max(execution_times),
            "total_bytes_processed": sum(bytes_processed),
            "cache_hit_rate": sum(1 for q in self.queries if q["cache_hit"])
            / len(self.queries),
        }


class MonitoredSQLExecutor(SQLQueryExecutor):
    """Wrapper that adds monitoring to any SQL executor"""

    def __init__(self, executor: SQLQueryExecutor):
        super().__init__(validator=executor.validator)
        self.executor = executor
        self.metrics = QueryMetrics()

    def execute_query(
        self, query: str, config: Optional[QueryConfig] = None
    ) -> QueryResult:
        """Execute with monitoring"""
        config = config or QueryConfig()
        result = self.executor.execute_query(query, config)
        self.metrics.record_query(query, result, config)
        return result

    async def execute_query_async(
        self, query: str, config: Optional[QueryConfig] = None
    ) -> QueryResult:
        """Execute async with monitoring"""
        config = config or QueryConfig()
        result = await self.executor.execute_query_async(query, config)
        self.metrics.record_query(query, result, config)
        return result

    def test_connection(self) -> bool:
        return self.executor.test_connection()

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get monitoring metrics"""
        return self.metrics.get_summary()


# Example usage with improved architecture
if __name__ == "__main__":
    import os

    # Create executor with monitoring
    base_executor = BigQuerySQLExecutor(project_id=os.getenv("GOOGLE_CLOUD_PROJECT"))
    executor = MonitoredSQLExecutor(base_executor)

    # Configure query execution
    config = QueryConfig(
        max_results=1000,
        timeout_seconds=30,
        use_cache=True,
        labels={"source": "research_agent", "env": "dev"},
    )

    # Execute query
    result = executor.execute_query(
        "SELECT COUNT(*) as total FROM `bigquery-public-data.samples.shakespeare`",
        config=config,
    )

    print(
        f"Query executed: {result.row_count} rows in {result.execution_time_seconds:.2f}s"
    )
    print(f"Cache hit: {result.metadata.get('cache_hit', False)}")
    print(f"Metrics: {executor.get_metrics_summary()}")
