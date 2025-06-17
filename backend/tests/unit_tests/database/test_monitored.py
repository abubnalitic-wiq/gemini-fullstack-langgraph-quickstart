"""Unit tests for monitored SQL executor"""

import pytest
from unittest.mock import Mock

from src.query_tools.monitoring import MonitoredSQLExecutor, QueryMetrics
from src.query_tools.database import QueryResult, QueryConfig, SQLQueryExecutor


class TestQueryMetrics:
    """Test query metrics collection"""

    def test_record_query(self) -> None:
        """Test recording query metrics"""
        metrics = QueryMetrics()

        result = QueryResult(
            rows=[{"id": 1}],
            column_names=["id"],
            row_count=1,
            execution_time_seconds=0.5,
            query="SELECT id FROM users",
            metadata={"bytes_processed": 1024, "cache_hit": True},
        )

        config = QueryConfig()
        metrics.record_query("SELECT id FROM users", result, config)

        assert len(metrics.queries) == 1
        metric = metrics.queries[0]
        assert metric["query"] == "SELECT id FROM users"
        assert metric["execution_time"] == 0.5

    def test_get_summary_empty(self) -> None:
        """Test summary with no queries"""
        metrics = QueryMetrics()
        summary = metrics.get_summary()

        assert summary == {"total_queries": 0}


class TestMonitoredSQLExecutor:
    """Test the monitored SQL executor wrapper"""

    @pytest.fixture
    def mock_executor(self) -> Mock:
        """Create a mock base executor"""
        return Mock(spec=SQLQueryExecutor)

    def test_initialization(self, mock_executor: Mock) -> None:
        """Test monitored executor initialization"""
        mock_executor.validator = Mock()
        monitored = MonitoredSQLExecutor(mock_executor)

        assert monitored.executor is mock_executor
        assert isinstance(monitored.metrics, QueryMetrics)

    def test_execute_query_with_monitoring(self, mock_executor: Mock) -> None:
        """Test that queries are monitored"""
        result = QueryResult(
            rows=[{"id": 1}],
            column_names=["id"],
            row_count=1,
            execution_time_seconds=0.5,
            query="SELECT id FROM users",
            metadata={"bytes_processed": 1024},
        )

        mock_executor.execute_query.return_value = result
        mock_executor.validator = Mock()

        monitored = MonitoredSQLExecutor(mock_executor)
        returned_result = monitored.execute_query("SELECT id FROM users")

        # Verify the result is passed through
        assert returned_result is result

        # Verify monitoring was recorded
        assert len(monitored.metrics.queries) == 1
