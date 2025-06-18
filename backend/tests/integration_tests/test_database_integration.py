"""Integration tests for database functionality"""

import pytest
import os
from typing import Optional

from src.query_tools.bigquery_connector import BigQuerySQLExecutor
from src.query_tools.database import QueryConfig


@pytest.mark.only_extended
class TestBigQueryIntegration:
    """Integration tests that require actual BigQuery access"""

    @pytest.fixture
    def project_id(self) -> Optional[str]:
        """Get project ID from environment"""
        return os.getenv("GOOGLE_CLOUD_PROJECT")

    @pytest.fixture
    def executor(self, project_id: Optional[str]) -> BigQuerySQLExecutor:
        """Create a real BigQuery executor"""
        if not project_id:
            pytest.skip("GOOGLE_CLOUD_PROJECT not set")

        return BigQuerySQLExecutor(project_id=project_id)

    def test_connection(self, executor: BigQuerySQLExecutor) -> None:
        """Test real BigQuery connection"""
        assert executor.test_connection() is True

    def test_simple_query(self, executor: BigQuerySQLExecutor) -> None:
        """Test executing a simple query"""
        config = QueryConfig(max_results=1)
        result = executor.execute_query("SELECT 1 as test_value", config)

        assert result.row_count == 1
        assert result.rows[0]["test_value"] == 1

    def test_public_dataset_query(self, executor: BigQuerySQLExecutor) -> None:
        """Test querying a public dataset"""
        config = QueryConfig(max_results=5)
        result = executor.execute_query(
            "SELECT word, word_count FROM `bigquery-public-data.samples.shakespeare` "
            "ORDER BY word_count DESC LIMIT 5",
            config,
        )

        assert result.row_count == 5
        assert "word" in result.column_names
        assert "word_count" in result.column_names

    @pytest.mark.asyncio
    async def test_async_query(self, executor: BigQuerySQLExecutor) -> None:
        """Test async query execution"""
        config = QueryConfig(max_results=1)
        result = await executor.execute_query_async("SELECT 1 as test_value", config)

        assert result.row_count == 1
        assert result.rows[0]["test_value"] == 1
