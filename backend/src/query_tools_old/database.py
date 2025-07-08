import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, List, Optional, Union


@dataclass
class QueryConfig:
    """Configuration for query execution."""

    max_results: int = 10000
    timeout_seconds: int = 30
    dry_run: bool = False
    use_cache: bool = True
    labels: Optional[Dict[str, str]] = None


@dataclass
class QueryResult:
    """Results from query execution."""

    rows: list[dict[str, Any]]
    column_names: list[str]
    row_count: int
    execution_time_seconds: float
    query: str
    is_truncated: bool = False
    total_rows_available: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])


class DatabaseError(Exception):
    """Base database error."""

    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        self.message = message
        self.query = query
        self.original_error = original_error
        super().__init__(self.message)


class QueryValidationError(DatabaseError):
    """Query validation specific error."""

    pass


class QueryValidator:
    """Extensible query validation."""

    def validate(self, query: str, config: QueryConfig) -> None:
        """Validate query - override for specific database rules."""
        if not query or not query.strip():
            raise QueryValidationError("Query cannot be empty")

        query_upper = query.upper().strip()

        # Block dangerous keywords
        dangerous_keywords = [
            "DROP",
            "DELETE",
            "UPDATE",
            "INSERT",
            "ALTER",
            "CREATE",
            "TRUNCATE",
        ]
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                raise QueryValidationError(
                    f"Query contains forbidden keyword: {keyword}"
                )


class SQLQueryExecutor(ABC):
    """Simplified SQL query executor interface.

    Provides both sync and async execution methods.
    """

    def __init__(self, validator: Optional[QueryValidator] = None):
        self.validator = validator or QueryValidator()

    @abstractmethod
    def execute_query(
        self, query: str, config: Optional[QueryConfig] = None
    ) -> QueryResult:
        """Execute query synchronously."""
        pass

    async def execute_query_async(
        self, query: str, config: Optional[QueryConfig] = None
    ) -> QueryResult:
        """Execute query asynchronously - default implementation runs in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute_query, query, config)

    @abstractmethod
    def test_connection(self) -> bool:
        """Test database connection."""
        pass

    def validate_query(self, query: str, config: Optional[QueryConfig] = None) -> None:
        """Validate query using configured validator."""
        self.validator.validate(query, config or QueryConfig())
