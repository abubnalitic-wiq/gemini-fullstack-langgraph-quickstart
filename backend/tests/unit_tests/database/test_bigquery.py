"""Unit tests for BigQuery SQL executor"""

import pytest
from unittest.mock import Mock, MagicMock
import asyncio
from google.cloud import bigquery
from google.cloud.exceptions import GoogleCloudError

from src.query_tools.bigquery_connector import BigQueryValidator, BigQuerySQLExecutor
from src.query_tools.database import (
    QueryConfig,
    QueryValidationError,
    DatabaseError,
)


class TestBigQueryValidator:
    """Test BigQuery-specific validation"""

    def test_cross_join_blocked(self) -> None:
        """Test that CROSS JOIN queries are blocked"""
        validator = BigQueryValidator()

        with pytest.raises(
            QueryValidationError, match="CROSS JOIN operations are not allowed"
        ):
            validator.validate("SELECT * FROM table1 CROSS JOIN table2", QueryConfig())

    def test_select_star_without_limit_blocked(self) -> None:
        """Test that SELECT * without LIMIT is blocked"""
        validator = BigQueryValidator()

        with pytest.raises(
            QueryValidationError, match="SELECT \\* queries must include LIMIT"
        ):
            validator.validate("SELECT * FROM users", QueryConfig())

    def test_select_star_with_limit_allowed(self) -> None:
        """Test that SELECT * with LIMIT is allowed"""
        validator = BigQueryValidator()

        # Should not raise exception
        validator.validate("SELECT * FROM users LIMIT 100", QueryConfig())


class TestBigQuerySQLExecutor:
    """Test BigQuery SQL executor"""

    @pytest.fixture
    def mock_client(self) -> Mock:
        """Create a mock BigQuery client"""
        return Mock(spec=bigquery.Client)

    @pytest.fixture
    def mock_query_job(self) -> Mock:
        """Create a mock query job"""
        job = Mock(spec=bigquery.QueryJob)
        job.job_id = "test_job_123"
        job.total_bytes_processed = 1024
        job.total_bytes_billed = 1024
        job.slot_millis = 100
        job.cache_hit = False
        return job

    @pytest.fixture
    def mock_query_results(self) -> Mock:
        """Create mock query results"""
        results = Mock()
        results.total_rows = 2
        id_field = Mock()
        id_field.name = "id"
        name_field = Mock()
        name_field.name = "name"
        results.schema = [id_field, name_field]
        results.__iter__ = Mock(return_value=iter([(1, "Alice"), (2, "Bob")]))
        return results

    def test_initialization(self) -> None:
        """Test executor initialization"""
        executor = BigQuerySQLExecutor(
            project_id="test-project", credentials_path="/path/to/creds.json"
        )

        assert executor.project_id == "test-project"
        assert executor.credentials_path == "/path/to/creds.json"
        assert isinstance(executor.validator, BigQueryValidator)

    def test_successful_query_execution(
        self, mock_client: Mock, mock_query_job: Mock, mock_query_results: Mock
    ) -> None:
        """Test successful query execution"""
        # Setup mocks
        mock_client.query.return_value = mock_query_job
        mock_query_job.result.return_value = mock_query_results

        # Create executor with mock client
        executor = BigQuerySQLExecutor(client_factory=lambda: mock_client)

        # Execute query
        result = executor.execute_query("SELECT id, name FROM users LIMIT 10")

        # Verify results
        assert result.row_count == 2
        assert result.column_names == ["id", "name"]
        assert result.rows == [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        assert result.query == "SELECT id, name FROM users LIMIT 10"
        assert result.metadata["job_id"] == "test_job_123"

    def test_bigquery_error_handling(self, mock_client: Mock) -> None:
        """Test BigQuery error handling"""
        mock_client.query.side_effect = GoogleCloudError("Table not found")

        executor = BigQuerySQLExecutor(client_factory=lambda: mock_client)

        with pytest.raises(DatabaseError, match="BigQuery error"):
            executor.execute_query("SELECT * FROM nonexistent_table LIMIT 10")

    @pytest.mark.asyncio
    async def test_async_query_execution(
        self, mock_client: Mock, mock_query_job: Mock, mock_query_results: Mock
    ) -> None:
        """Test async query execution"""
        mock_client.query.return_value = mock_query_job
        mock_query_job.result.return_value = mock_query_results

        executor = BigQuerySQLExecutor(client_factory=lambda: mock_client)

        result = await executor.execute_query_async(
            "SELECT id, name FROM users LIMIT 10"
        )

        assert result.row_count == 2
        assert result.column_names == ["id", "name"]
