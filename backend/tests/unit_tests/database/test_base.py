"""Unit tests for the base SQL query executor components"""

from typing import Optional
from unittest.mock import Mock

import pytest

from src.query_tools.database import (
    DatabaseError,
    QueryConfig,
    QueryResult,
    QueryValidationError,
    QueryValidator,
    SQLQueryExecutor,
)


class TestQueryValidator:
    """Test the base query validator"""

    def test_valid_select_query(self) -> None:
        """Test that valid SELECT queries pass validation"""
        validator = QueryValidator()

        # Should not raise any exception
        validator.validate("SELECT * FROM users LIMIT 10", QueryConfig())
        validator.validate("SELECT id, name FROM products", QueryConfig())
        validator.validate("  select count(*) from orders  ", QueryConfig())

    def test_empty_query_fails(self) -> None:
        """Test that empty queries fail validation"""
        validator = QueryValidator()

        with pytest.raises(QueryValidationError, match="Query cannot be empty"):
            validator.validate("", QueryConfig())

        with pytest.raises(QueryValidationError, match="Query cannot be empty"):
            validator.validate("   ", QueryConfig())

    def test_non_select_queries_fail(self) -> None:
        """Test that non-SELECT queries fail validation"""
        validator = QueryValidator()

        dangerous_queries = [
            "DROP TABLE users",
            "DELETE FROM users WHERE id = 1",
            "UPDATE users SET name = 'test'",
            "INSERT INTO users (name) VALUES ('test')",
        ]

        for query in dangerous_queries:
            with pytest.raises(QueryValidationError):
                validator.validate(query, QueryConfig())


class TestQueryConfig:
    """Test query configuration"""

    def test_default_config(self) -> None:
        """Test default configuration values"""
        config = QueryConfig()

        assert config.max_results == 10000
        assert config.timeout_seconds == 30
        assert config.dry_run is False
        assert config.use_cache is True
        assert config.labels is None

    def test_custom_config(self) -> None:
        """Test custom configuration values"""
        config = QueryConfig(
            max_results=5000,
            timeout_seconds=60,
            dry_run=True,
            use_cache=False,
            labels={"env": "test"},
        )

        assert config.max_results == 5000
        assert config.timeout_seconds == 60
        assert config.dry_run is True
        assert config.use_cache is False
        assert config.labels == {"env": "test"}


class TestQueryResult:
    """Test query result data structure"""

    def test_basic_result(self) -> None:
        """Test basic query result creation"""
        result = QueryResult(
            rows=[{"id": 1, "name": "test"}],
            column_names=["id", "name"],
            row_count=1,
            execution_time_seconds=0.5,
            query="SELECT id, name FROM users",
        )

        assert len(result.rows) == 1
        assert result.rows[0]["name"] == "test"
        assert result.column_names == ["id", "name"]
        assert result.row_count == 1
        assert result.execution_time_seconds == 0.5
        assert result.is_truncated is False
        assert result.total_rows_available is None
