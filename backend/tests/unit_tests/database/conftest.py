"""Shared test fixtures for database tests"""

from typing import Any, Dict, List
from unittest.mock import Mock

import pytest

from src.query_tools.database import QueryConfig, QueryResult


@pytest.fixture
def sample_query_result() -> QueryResult:
    """Fixture providing a sample query result"""
    return QueryResult(
        rows=[
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": "Bob", "age": 25},
        ],
        column_names=["id", "name", "age"],
        row_count=2,
        execution_time_seconds=0.5,
        query="SELECT id, name, age FROM users",
        metadata={
            "job_id": "test_job_123",
            "bytes_processed": 1024,
            "cache_hit": False,
        },
    )


@pytest.fixture
def sample_config() -> QueryConfig:
    """Fixture providing a sample query config"""
    return QueryConfig(
        max_results=1000,
        timeout_seconds=30,
        dry_run=False,
        use_cache=True,
        labels={"test": "true"},
    )


def create_mock_query_result(
    rows: List[Dict[str, Any]],
    query: str = "SELECT * FROM test",
    execution_time: float = 0.5,
) -> QueryResult:
    """Helper to create mock query results"""
    if not rows:
        return QueryResult(
            rows=[],
            column_names=[],
            row_count=0,
            execution_time_seconds=execution_time,
            query=query,
        )

    column_names = list(rows[0].keys())

    return QueryResult(
        rows=rows,
        column_names=column_names,
        row_count=len(rows),
        execution_time_seconds=execution_time,
        query=query,
        metadata={
            "job_id": "mock_job_123",
            "bytes_processed": 1024 * len(rows),
            "cache_hit": False,
        },
    )
