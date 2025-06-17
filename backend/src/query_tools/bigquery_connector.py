"""This module provides classes and utilities for executing and validating SQL queries against Google BigQuery.

It includes:

- `BigQueryValidator`: Implements BigQuery-specific query validation rules, such as restricting CROSS JOINs and enforcing LIMIT on SELECT * queries.
- `BigQuerySQLExecutor`: Handles synchronous and asynchronous execution of SQL queries on BigQuery, manages client initialization, error handling, and result formatting.
- Utility methods for testing BigQuery connections and retrieving table schema information.

Dependencies:
- google-cloud-bigquery
- Custom database abstractions: DatabaseError, QueryConfig, QueryResult, QueryValidationError, QueryValidator, SQLQueryExecutor

Intended for use in backend services requiring secure, validated, and efficient interaction with BigQuery.
"""

import asyncio
import time
from typing import Callable, Dict, List, Optional

from google.cloud import bigquery
from google.cloud.exceptions import GoogleCloudError
from loguru import logger

from src.query_tools.database import (
    DatabaseError,
    QueryConfig,
    QueryResult,
    QueryValidationError,
    QueryValidator,
    SQLQueryExecutor,
)


class BigQueryValidator(QueryValidator):
    """BigQuery-specific validation rules."""

    def validate(self, query: str, config: QueryConfig) -> None:
        # Run base validation first
        super().validate(query, config)

        query_upper = query.upper().strip()

        # BigQuery-specific validations
        if "CROSS JOIN" in query_upper:
            raise QueryValidationError(
                "CROSS JOIN operations are not allowed due to potential high costs"
            )

        # Check for expensive wildcard operations
        if "SELECT *" in query_upper and "LIMIT" not in query_upper:
            raise QueryValidationError("SELECT * queries must include LIMIT clause")


class BigQuerySQLExecutor(SQLQueryExecutor):
    """Simplified BigQuery SQL executor."""

    def __init__(
        self,
        project_id: Optional[str] = None,
        credentials_path: Optional[str] = None,
        client_factory: Optional[Callable] = None,
    ):
        """Initialize BigQuery executor.

        Args:
            project_id: GCP project ID
            credentials_path: Path to service account credentials
            client_factory: Factory function for creating client (useful for testing)
        """
        super().__init__(validator=BigQueryValidator())

        self.project_id = project_id
        self.credentials_path = credentials_path
        self.client_factory = client_factory or self._default_client_factory
        self._client: Optional[bigquery.Client] = None

    def _default_client_factory(self) -> bigquery.Client:
        if self.credentials_path:
            logger.debug(
                f"Initializing BigQuery client with credentials_path={self.credentials_path}, project_id={self.project_id}"
            )
            return bigquery.Client.from_service_account_json(
                self.credentials_path, project=self.project_id
            )
        else:
            return bigquery.Client(project=self.project_id)

    @property
    def client(self) -> bigquery.Client:
        """Lazy client initialization."""
        if self._client is None:
            try:
                self._client = self.client_factory()
            except Exception as e:
                raise DatabaseError(
                    f"Failed to initialize BigQuery client: {str(e)}", original_error=e
                )
        return self._client

    def execute_query(
        self, query: str, config: Optional[QueryConfig] = None
    ) -> QueryResult:
        """Execute BigQuery SQL query."""
        config = config or QueryConfig()
        start_time = time.time()

        # Validate query
        self.validate_query(query, config)

        try:
            # Configure job
            job_config = bigquery.QueryJobConfig(
                dry_run=config.dry_run,
                use_query_cache=config.use_cache,
                labels=config.labels or {},
            )

            # Execute query
            query_job = self.client.query(
                query, job_config=job_config, timeout=config.timeout_seconds
            )

            if config.dry_run:
                # For dry run, return empty result with metadata
                return QueryResult(
                    rows=[],
                    column_names=[],
                    row_count=0,
                    execution_time_seconds=time.time() - start_time,
                    query=query,
                    metadata={
                        "dry_run": True,
                        "bytes_processed": query_job.total_bytes_processed,
                        "job_id": query_job.job_id,
                    },
                )

            # Get results with limit
            results = query_job.result(max_results=config.max_results)

            # Convert to list of dicts (more memory efficient than pandas for small results)
            rows = []
            column_names = []

            if results.total_rows > 0:
                # Get column names from schema
                column_names = [field.name for field in results.schema]

                # Convert rows to dictionaries
                for row in results:
                    rows.append(dict(zip(column_names, row)))

            execution_time = time.time() - start_time
            is_truncated = (
                len(rows) >= config.max_results
                and results.total_rows > config.max_results
            )

            return QueryResult(
                rows=rows,
                column_names=column_names,
                row_count=len(rows),
                execution_time_seconds=execution_time,
                query=query,
                is_truncated=is_truncated,
                total_rows_available=results.total_rows,
                metadata={
                    "job_id": query_job.job_id,
                    "bytes_processed": query_job.total_bytes_processed,
                    "bytes_billed": query_job.total_bytes_billed,
                    "slot_millis": query_job.slot_millis,
                    "project_id": self.project_id,
                    "cache_hit": query_job.cache_hit,
                },
            )

        except GoogleCloudError as e:
            raise DatabaseError(
                f"BigQuery error: {str(e)}", query=query, original_error=e
            )
        except Exception as e:
            raise DatabaseError(
                f"Unexpected error: {str(e)}", query=query, original_error=e
            )

    async def execute_query_async(
        self, query: str, config: Optional[QueryConfig] = None
    ) -> QueryResult:
        """Execute query asynchronously.

        True async implementation for BigQuery.
        Uses asyncio for better performance than thread pool.
        """
        config = config or QueryConfig()

        # Validate query
        self.validate_query(query, config)

        # Use asyncio to run in thread pool (BigQuery client doesn't have native async support)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute_query, query, config)

    def test_connection(self) -> bool:
        """Test BigQuery connection."""
        try:
            test_config = QueryConfig(max_results=1, timeout_seconds=10)
            result = self.execute_query("SELECT 1 as test_value", test_config)
            return result.row_count == 1 and result.rows[0]["test_value"] == 1
        except Exception as e:
            print(f"Error testing connection: {e}")
            return False

    def get_table_schema(self, table_id: str) -> list[dict[str, str]]:
        """Get table schema information."""
        try:
            table = self.client.get_table(table_id)
            return [
                {
                    "name": field.name,
                    "type": field.field_type,
                    "mode": field.mode,
                    "description": field.description or "",
                }
                for field in table.schema
            ]
        except Exception as e:
            raise DatabaseError(
                f"Failed to get table schema: {str(e)}", original_error=e
            )

    def list_datasets(self) -> list[str]:
        """List all datasets in the BigQuery project."""
        try:
            datasets = self.client.list_datasets()
            return [dataset.dataset_id for dataset in datasets]
        except Exception as e:
            raise DatabaseError(f"Failed to list datasets: {str(e)}", original_error=e)

    def list_tables(self, dataset_id: str) -> list[str]:
        """List all tables in a BigQuery dataset."""
        try:
            tables = self.client.list_tables(dataset_id)
            return [table.table_id for table in tables]
        except Exception as e:
            raise DatabaseError(f"Failed to list tables: {str(e)}", original_error=e)
