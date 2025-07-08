"""Query and Result Formatters for LLM Integration.

This module provides formatters optimized for LLM ingestion and analysis of SQL queries,
query results, and parameters. All formatting is designed to maximize LLM understanding
while minimizing token usage.

Key Features:
- LLM-optimized SQL formatting with semantic structure
- Business context injection for better LLM understanding
- Multiple output formats for parameters and results
- Query analysis prompt generation
- Business logic annotation capabilities
"""

import csv
import json
import re
from datetime import datetime
from io import StringIO
from typing import Any, Dict, List, Optional, TypeAlias, Union

import pandas as pd
from loguru import logger
from tabulate import tabulate

from src.query_tools.database import QueryResult

# Type aliases
QueryParams: TypeAlias = Dict[str, Any]
FormattedOutput: TypeAlias = str
FormatStyle: TypeAlias = str


class QueryFormatterError(Exception):
    """Raised when query formatting fails."""

    pass


class SQLQueryFormatter:
    """Formats SQL queries optimized for LLM ingestion and analysis."""

    def __init__(self, indent_size: int = 2):
        """Initialize SQL formatter.

        Args:
            indent_size: Number of spaces for indentation
        """
        self.indent_size = indent_size

    def format_sql(self, query: str) -> str:
        """Format SQL query optimized for LLM understanding.

        Args:
            query: Raw SQL query string

        Returns:
            LLM-optimized formatted SQL query string

        Example:
            >>> formatter = SQLQueryFormatter()
            >>> formatted = formatter.format_sql("SELECT * FROM table WHERE id = 1")
        """
        try:
            return self._format_for_llm(query)

        except Exception as e:
            logger.warning(f"SQL formatting failed: {e}, returning original query")
            return query

    def _format_for_llm(self, query: str) -> str:
        """Format SQL specifically for LLM ingestion with the following.

        - Compact but clear structure
        - Consistent patterns
        - Semantic clarity
        - Parameter highlighting.
        """
        # Normalize whitespace
        query = re.sub(r"\s+", " ", query.strip())

        # LLM-specific formatting
        formatted = self._llm_format_structure(query)
        formatted = self._llm_highlight_parameters(formatted)
        formatted = self._llm_format_functions(formatted)
        formatted = self._llm_format_conditions(formatted)
        formatted = self._llm_cleanup(formatted)

        return formatted

    def _llm_format_structure(self, query: str) -> str:
        """Format major SQL structure for LLM understanding."""
        # Major keywords with consistent spacing
        major_keywords = [
            "SELECT",
            "FROM",
            "WHERE",
            "GROUP BY",
            "HAVING",
            "ORDER BY",
            "LIMIT",
            "UNION",
            "DECLARE",
            "SET",
        ]

        for keyword in major_keywords:
            pattern = rf"\b{keyword}\b"
            query = re.sub(pattern, f"\n{keyword}", query, flags=re.IGNORECASE)

        # Format SELECT columns with consistent pattern
        query = self._llm_format_select_columns(query)

        # Format JOINs consistently
        join_keywords = [
            "JOIN",
            "LEFT JOIN",
            "RIGHT JOIN",
            "INNER JOIN",
            "OUTER JOIN",
            "FULL JOIN",
        ]
        for join_type in join_keywords:
            pattern = rf"\b{join_type}\b"
            query = re.sub(pattern, f"\n  {join_type}", query, flags=re.IGNORECASE)

        return query

    def _llm_format_select_columns(self, query: str) -> str:
        """Format SELECT columns for LLM - more compact than human format."""

        def format_select_block(match):
            select_part = match.group(0)
            columns = self._split_on_commas_outside_parens(select_part)

            if len(columns) <= 1:
                return select_part

            # More compact formatting for LLM
            formatted_columns = []
            for i, col in enumerate(columns):
                col = col.strip()
                if i == 0:
                    formatted_columns.append(col)
                else:
                    formatted_columns.append(f"  {col}")

            return ",\n".join(formatted_columns)

        pattern = r"SELECT\s+.*?(?=\s+FROM|\s+$)"
        return re.sub(
            pattern, format_select_block, query, flags=re.IGNORECASE | re.DOTALL
        )

    def _llm_highlight_parameters(self, query: str) -> str:
        """Highlight parameters for LLM attention."""
        # Make parameters more visible with consistent spacing
        query = re.sub(r"(@\w+)", r" \1 ", query)
        # Clean up extra spaces
        query = re.sub(r"\s+", " ", query)
        return query

    def _llm_format_functions(self, query: str) -> str:
        """Format SQL functions for LLM understanding."""
        # Format common aggregation functions
        agg_functions = ["SUM", "COUNT", "AVG", "MIN", "MAX", "ROUND"]
        for func in agg_functions:
            pattern = rf"\b{func}\s*\("
            query = re.sub(pattern, f"{func}(", query, flags=re.IGNORECASE)

        # Format date functions with clearer structure
        date_functions = ["DATE_SUB", "DATE_TRUNC", "CURRENT_DATE"]
        for func in date_functions:
            pattern = rf"\b{func}\s*\("
            query = re.sub(pattern, f"{func}(", query, flags=re.IGNORECASE)

        return query

    def _llm_format_conditions(self, query: str) -> str:
        """Format WHERE/HAVING conditions for LLM clarity."""
        # Format logical operators consistently
        query = re.sub(r"\s+(AND|OR)\s+", r"\n  \1 ", query, flags=re.IGNORECASE)

        # Format IN clauses
        query = re.sub(
            r"\bIN\s+UNNEST\s*\(\s*(@\w+)\s*\)",
            r"IN UNNEST(\1)",
            query,
            flags=re.IGNORECASE,
        )

        # Format comparison operators
        query = re.sub(r"\s*(=|!=|<>|<=|>=|<|>)\s*", r" \1 ", query)

        return query

    def _llm_cleanup(self, query: str) -> str:
        """Final cleanup for LLM formatting."""
        # Remove excessive blank lines but keep structure
        query = re.sub(r"\n\s*\n\s*\n", "\n\n", query)

        # Ensure consistent spacing around keywords
        query = re.sub(r"\n\s*([A-Z]+)\s+", r"\n\1 ", query)

        # Clean up trailing spaces
        lines = [line.rstrip() for line in query.split("\n")]
        query = "\n".join(lines).strip()

        return query

    def _format_major_clauses(self, query: str) -> str:
        """Format major SQL clauses (SELECT, FROM, WHERE, etc.)."""
        # Major SQL keywords that should start new lines
        major_keywords = [
            "SELECT",
            "FROM",
            "WHERE",
            "GROUP BY",
            "HAVING",
            "ORDER BY",
            "LIMIT",
            "UNION",
            "DECLARE",
            "SET",
        ]

        for keyword in major_keywords:
            # Add line breaks before major keywords
            pattern = rf"\b{keyword}\b"
            query = re.sub(pattern, f"\n{keyword}", query, flags=re.IGNORECASE)

        return query

    def _format_select_columns(self, query: str) -> str:
        """Format SELECT column list."""

        def format_select_block(match):
            select_part = match.group(0)

            # Split on commas that aren't inside parentheses
            columns = self._split_on_commas_outside_parens(select_part)

            if len(columns) <= 1:
                return select_part

            # Format each column
            formatted_columns = []
            for i, col in enumerate(columns):
                col = col.strip()
                if i == 0:
                    # First column stays on SELECT line
                    formatted_columns.append(col)
                else:
                    # Subsequent columns are indented
                    formatted_columns.append(f"{' ' * self.indent_size}{col}")

            return ",\n".join(formatted_columns)

        # Match SELECT clause until FROM
        pattern = r"SELECT\s+.*?(?=\s+FROM|\s+$)"
        return re.sub(
            pattern, format_select_block, query, flags=re.IGNORECASE | re.DOTALL
        )

    def _format_joins(self, query: str) -> str:
        """Format JOIN clauses."""
        join_keywords = [
            "JOIN",
            "LEFT JOIN",
            "RIGHT JOIN",
            "INNER JOIN",
            "OUTER JOIN",
            "FULL JOIN",
        ]

        for join_type in join_keywords:
            pattern = rf"\b{join_type}\b"
            query = re.sub(
                pattern,
                f'\n{" " * self.indent_size}{join_type}',
                query,
                flags=re.IGNORECASE,
            )

        return query

    def _format_where_conditions(self, query: str) -> str:
        """Format WHERE conditions."""
        # Add line breaks before AND/OR in WHERE clauses
        query = re.sub(
            r"\s+(AND|OR)\s+",
            r"\n" + " " * (self.indent_size * 2) + r"\1 ",
            query,
            flags=re.IGNORECASE,
        )

        return query

    def _format_case_statements(self, query: str) -> str:
        """Format CASE statements."""

        def format_case_block(match):
            case_content = match.group(0)

            # Format WHEN clauses
            case_content = re.sub(
                r"\s+WHEN\s+",
                f'\n{" " * (self.indent_size * 2)}WHEN ',
                case_content,
                flags=re.IGNORECASE,
            )
            case_content = re.sub(
                r"\s+THEN\s+",
                f'\n{" " * (self.indent_size * 3)}THEN ',
                case_content,
                flags=re.IGNORECASE,
            )
            case_content = re.sub(
                r"\s+ELSE\s+",
                f'\n{" " * (self.indent_size * 2)}ELSE ',
                case_content,
                flags=re.IGNORECASE,
            )
            case_content = re.sub(
                r"\s+END\b",
                f'\n{" " * self.indent_size}END',
                case_content,
                flags=re.IGNORECASE,
            )

            return case_content

        # Match CASE statements
        pattern = r"CASE\b.*?\bEND"
        return re.sub(
            pattern, format_case_block, query, flags=re.IGNORECASE | re.DOTALL
        )

    def _split_on_commas_outside_parens(self, text: str) -> List[str]:
        """Split text on commas that are outside parentheses."""
        parts = []
        current_part = ""
        paren_depth = 0

        for char in text:
            if char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth -= 1
            elif char == "," and paren_depth == 0:
                parts.append(current_part)
                current_part = ""
                continue

            current_part += char

        if current_part:
            parts.append(current_part)

        return parts

    def _final_cleanup(self, query: str) -> str:
        """Cleanup formatted query."""
        # Remove extra blank lines
        query = re.sub(r"\n\s*\n", "\n", query)

        # Ensure proper spacing after keywords
        query = re.sub(r"(\w)\n(\w)", r"\1\n\n\2", query)

        # Clean up leading/trailing whitespace
        lines = [line.rstrip() for line in query.split("\n")]
        query = "\n".join(lines).strip()

        return query


class LLMQueryFormatter:
    """Specialized formatter for preparing SQL queries for LLM analysis and generation."""

    def __init__(self):
        """Initialize LLM query formatter."""
        self.sql_formatter = SQLQueryFormatter()

    def format_query_for_llm(
        self,
        query: str,
        params: QueryParams,
        business_context: Optional[str] = None,
        table_descriptions: Optional[Dict[str, str]] = None,
        column_descriptions: Optional[Dict[str, str]] = None,
    ) -> str:
        """Format query with full context for LLM ingestion.

        Args:
            query: Raw SQL query
            params: Query parameters
            business_context: Business purpose description
            table_descriptions: Dict of table_name -> description
            column_descriptions: Dict of column_name -> description

        Returns:
            Formatted query with full context for LLM

        Example:
            >>> formatter = LLMQueryFormatter()
            >>> llm_query = formatter.format_query_for_llm(
            ...     query=sql_query,
            ...     params=query_params,
            ...     business_context="Weekly store performance analysis",
            ...     table_descriptions={"fact_profit": "Store financial performance data"}
            ... )
        """
        sections = []

        # Business Context Header
        sections.append("-- BUSINESS CONTEXT")
        if business_context:
            sections.append(f"-- Purpose: {business_context}")
        else:
            sections.append("-- Purpose: Retail analytics query")
        sections.append("")

        # Parameter Context
        if params:
            sections.append("-- PARAMETERS")
            for param_name, param_value in params.items():
                param_type = type(param_value).__name__
                if isinstance(param_value, list):
                    value_desc = (
                        f"Array of {len(param_value)} {param_type}s: {param_value}"
                    )
                else:
                    value_desc = f"{param_type}: {param_value}"
                sections.append(f"-- @{param_name} = {value_desc}")
            sections.append("")

        # Table Context
        if table_descriptions:
            sections.append("-- TABLE DESCRIPTIONS")
            for table_name, description in table_descriptions.items():
                sections.append(f"-- {table_name}: {description}")
            sections.append("")

        # Column Context (if provided)
        if column_descriptions:
            sections.append("-- KEY COLUMN DESCRIPTIONS")
            for column_name, description in column_descriptions.items():
                sections.append(f"-- {column_name}: {description}")
            sections.append("")

        # Formatted SQL
        sections.append("-- SQL QUERY")
        formatted_sql = self.sql_formatter.format_sql(query, for_llm=True)
        sections.append(formatted_sql)

        return "\n".join(sections)

    def create_query_analysis_prompt(
        self, query: str, params: QueryParams, analysis_type: str = "explain"
    ) -> str:
        """Create a structured prompt for LLM query analysis.

        Args:
            query: SQL query to analyze
            params: Query parameters
            analysis_type: Type of analysis ("explain", "optimize", "validate")

        Returns:
            Structured prompt for LLM analysis
        """
        formatted_query = self.format_query_for_llm(query, params)

        if analysis_type == "explain":
            prompt_header = """Please analyze this SQL query and explain:
1. What business question it answers
2. Key data transformations being performed
3. Important filters and aggregations
4. Potential performance considerations

Query to analyze:"""

        elif analysis_type == "optimize":
            prompt_header = """Please review this SQL query for optimization opportunities:
1. Index recommendations
2. Query structure improvements
3. Potential performance bottlenecks
4. Alternative approaches

Query to optimize:"""

        elif analysis_type == "validate":
            prompt_header = """Please validate this SQL query for:
1. Syntax correctness
2. Logical consistency
3. Potential data quality issues
4. Security considerations (parameter usage)

Query to validate:"""

        else:
            prompt_header = (
                f"Please analyze this SQL query for {analysis_type}:\n\nQuery:"
            )

        return f"{prompt_header}\n\n{formatted_query}"

    def create_query_generation_context(
        self,
        business_requirement: str,
        available_tables: Dict[str, str],
        key_columns: Dict[str, str],
        sample_parameters: QueryParams,
    ) -> str:
        """Create context for LLM to generate similar queries.

        Args:
            business_requirement: What the query should accomplish
            available_tables: Dict of table_name -> description
            key_columns: Dict of column_name -> description
            sample_parameters: Example parameters to use

        Returns:
            Structured context for query generation
        """
        sections = []

        sections.append("-- QUERY GENERATION CONTEXT")
        sections.append(f"-- Business Requirement: {business_requirement}")
        sections.append("")

        sections.append("-- AVAILABLE TABLES")
        for table_name, description in available_tables.items():
            sections.append(f"-- {table_name}: {description}")
        sections.append("")

        sections.append("-- KEY COLUMNS")
        for column_name, description in key_columns.items():
            sections.append(f"-- {column_name}: {description}")
        sections.append("")

        sections.append("-- PARAMETER EXAMPLES")
        for param_name, param_value in sample_parameters.items():
            sections.append(f"-- @{param_name} = {param_value}")
        sections.append("")

        sections.append("-- REQUIREMENTS")
        sections.append("-- 1. Use parameterized queries with @parameter syntax")
        sections.append("-- 2. Include proper aggregations and grouping")
        sections.append("-- 3. Apply appropriate filters and date logic")
        sections.append("-- 4. Follow BigQuery SQL syntax")
        sections.append("")

        return "\n".join(sections)

    def annotate_query_with_business_logic(
        self, query: str, annotations: Dict[str, str]
    ) -> str:
        """Add business logic annotations to query sections.

        Args:
            query: SQL query to annotate
            annotations: Dict of pattern -> annotation

        Returns:
            Query with business logic annotations
        """
        formatted_query = self.sql_formatter.format_sql(query, for_llm=True)
        lines = formatted_query.split("\n")
        annotated_lines = []

        for line in lines:
            annotated_lines.append(line)

            # Check for annotation patterns
            for pattern, annotation in annotations.items():
                if re.search(pattern, line, re.IGNORECASE):
                    annotated_lines.append(f"  -- Business Logic: {annotation}")

        return "\n".join(annotated_lines)


class ParameterFormatter:
    """Formats query parameters for display and logging."""

    def format_parameters(self, params: QueryParams, style: str = "table") -> str:
        """Format query parameters in various styles.

        Args:
            params: Dictionary of parameter name-value pairs
            style: Format style ("table", "json", "inline", "yaml")

        Returns:
            Formatted parameter string

        Example:
            >>> formatter = ParameterFormatter()
            >>> formatted = formatter.format_parameters(
            ...     {"salesorg": "1005", "regions": ["NSW", "VIC"]},
            ...     style="table"
            ... )
        """
        if not params:
            return "No parameters"

        try:
            if style == "table":
                return self._format_as_table(params)
            elif style == "json":
                return self._format_as_json(params)
            elif style == "inline":
                return self._format_as_inline(params)
            elif style == "yaml":
                return self._format_as_yaml(params)
            else:
                raise QueryFormatterError(f"Unknown parameter format style: {style}")

        except Exception as e:
            logger.warning(f"Parameter formatting failed: {e}")
            return str(params)

    def _format_as_table(self, params: QueryParams) -> str:
        """Format parameters as a table."""
        table_data = []
        for key, value in params.items():
            # Format value based on type
            if isinstance(value, list):
                formatted_value = f"[{', '.join(map(str, value))}]"
            elif isinstance(value, str):
                formatted_value = f"'{value}'"
            else:
                formatted_value = str(value)

            table_data.append([key, type(value).__name__, formatted_value])

        return tabulate(
            table_data, headers=["Parameter", "Type", "Value"], tablefmt="grid"
        )

    def _format_as_json(self, params: QueryParams) -> str:
        """Format parameters as JSON."""
        return json.dumps(params, indent=2, default=str)

    def _format_as_inline(self, params: QueryParams) -> str:
        """Format parameters as inline string."""
        formatted_params = []
        for key, value in params.items():
            if isinstance(value, list):
                formatted_value = f"[{', '.join(map(str, value))}]"
            elif isinstance(value, str):
                formatted_value = f"'{value}'"
            else:
                formatted_value = str(value)

            formatted_params.append(f"{key}={formatted_value}")

        return ", ".join(formatted_params)

    def _format_as_yaml(self, params: QueryParams) -> str:
        """Format parameters as YAML-like string."""
        lines = []
        for key, value in params.items():
            if isinstance(value, list):
                lines.append(f"{key}:")
                for item in value:
                    lines.append(f"  - {item}")
            else:
                lines.append(f"{key}: {value}")

        return "\n".join(lines)


class ResultFormatter:
    """Formats query results in various output formats."""

    def __init__(self, max_display_rows: int = 100):
        """Initialize result formatter.

        Args:
            max_display_rows: Maximum number of rows to display in table format
        """
        self.max_display_rows = max_display_rows

    def format_result(
        self,
        result: QueryResult,
        format_type: str = "table",
        include_metadata: bool = True,
    ) -> str:
        """Format query result in specified format.

        Args:
            result: QueryResult object to format
            format_type: Output format ("table", "json", "csv", "summary")
            include_metadata: Whether to include query metadata

        Returns:
            Formatted result string

        Example:
            >>> formatter = ResultFormatter()
            >>> formatted = formatter.format_result(result, "table")
        """
        try:
            if format_type == "table":
                return self._format_as_table(result, include_metadata)
            elif format_type == "json":
                return self._format_as_json(result, include_metadata)
            elif format_type == "csv":
                return self._format_as_csv(result, include_metadata)
            elif format_type == "summary":
                return self._format_as_summary(result)
            elif format_type == "dataframe":
                return self._format_as_dataframe(result)
            else:
                raise QueryFormatterError(f"Unknown result format type: {format_type}")

        except Exception as e:
            logger.error(f"Result formatting failed: {e}")
            return f"Error formatting result: {str(e)}"

    def _format_as_table(self, result: QueryResult, include_metadata: bool) -> str:
        """Format result as a table."""
        output_lines = []

        # Add metadata header if requested
        if include_metadata:
            output_lines.append("=" * 80)
            output_lines.append("QUERY EXECUTION SUMMARY")
            output_lines.append("=" * 80)
            output_lines.append(f"Rows returned: {result.row_count:,}")
            if result.total_rows_available:
                output_lines.append(
                    f"Total rows available: {result.total_rows_available:,}"
                )
            output_lines.append(
                f"Execution time: {result.execution_time_seconds:.2f} seconds"
            )

            if result.metadata:
                if result.metadata.get("bytes_processed"):
                    bytes_processed = result.metadata["bytes_processed"]
                    output_lines.append(f"Bytes processed: {bytes_processed:,}")

                if result.metadata.get("cache_hit"):
                    output_lines.append("Cache hit: Yes")

                if result.is_truncated:
                    output_lines.append("⚠️  Results truncated - more data available")

            output_lines.append("=" * 80)
            output_lines.append("")

        # Format data table
        if result.rows:
            # Limit rows for display
            display_rows = result.rows[: self.max_display_rows]

            # Convert to tabulate format
            table_data = []
            for row in display_rows:
                if isinstance(row, dict):
                    table_data.append(
                        [str(row.get(col, "")) for col in result.column_names]
                    )
                else:
                    table_data.append([str(val) for val in row])

            table_output = tabulate(
                table_data,
                headers=result.column_names,
                tablefmt="grid",
                stralign="left",
            )
            output_lines.append(table_output)

            # Add truncation notice if needed
            if len(result.rows) > self.max_display_rows:
                output_lines.append("")
                output_lines.append(
                    f"... and {len(result.rows) - self.max_display_rows} more rows"
                )
        else:
            output_lines.append("No data returned")

        return "\n".join(output_lines)

    def _format_as_json(self, result: QueryResult, include_metadata: bool) -> str:
        """Format result as JSON."""
        output = {
            "data": result.rows,
            "column_names": result.column_names,
            "row_count": result.row_count,
        }

        if include_metadata:
            output["metadata"] = {
                "execution_time_seconds": result.execution_time_seconds,
                "is_truncated": result.is_truncated,
                "total_rows_available": result.total_rows_available,
                **result.metadata,
            }

        return json.dumps(output, indent=2, default=str)

    def _format_as_csv(self, result: QueryResult, include_metadata: bool) -> str:
        """Format result as CSV."""
        output = StringIO()

        if include_metadata and result.metadata:
            # Add metadata as comments
            output.write(f"# Query executed at: {datetime.now().isoformat()}\n")
            output.write(f"# Rows returned: {result.row_count}\n")
            output.write(
                f"# Execution time: {result.execution_time_seconds:.2f} seconds\n"
            )
            if result.metadata.get("bytes_processed"):
                output.write(
                    f"# Bytes processed: {result.metadata['bytes_processed']}\n"
                )
            output.write("#\n")

        if result.rows and result.column_names:
            writer = csv.DictWriter(output, fieldnames=result.column_names)
            writer.writeheader()

            for row in result.rows:
                if isinstance(row, dict):
                    writer.writerow(row)
                else:
                    # Convert list/tuple to dict
                    row_dict = dict(zip(result.column_names, row))
                    writer.writerow(row_dict)

        return output.getvalue()

    def _format_as_summary(self, result: QueryResult) -> str:
        """Format result as a summary."""
        lines = []
        lines.append(f"Query Result Summary")
        lines.append(f"==================")
        lines.append(f"Rows: {result.row_count:,}")
        lines.append(f"Columns: {len(result.column_names)}")
        lines.append(f"Execution Time: {result.execution_time_seconds:.2f}s")

        if result.metadata:
            if result.metadata.get("bytes_processed"):
                lines.append(f"Bytes Processed: {result.metadata['bytes_processed']:,}")
            if result.metadata.get("cache_hit"):
                lines.append(
                    f"Cache Hit: {'Yes' if result.metadata['cache_hit'] else 'No'}"
                )

        if result.column_names:
            lines.append(f"\nColumns: {', '.join(result.column_names)}")

        return "\n".join(lines)

    def _format_as_dataframe(self, result: QueryResult) -> str:
        """Format result as pandas DataFrame string representation."""
        if not result.rows:
            return "Empty DataFrame"

        try:
            df = pd.DataFrame(result.rows)
            return str(df)
        except Exception as e:
            return f"Error creating DataFrame: {str(e)}"


class QueryDisplayFormatter:
    """Formats complete query information optimized for LLM analysis."""

    def __init__(self):
        """Initialize query display formatter."""
        self.sql_formatter = SQLQueryFormatter()
        self.param_formatter = ParameterFormatter()
        self.result_formatter = ResultFormatter()
        self.llm_formatter = LLMQueryFormatter()

    def format_query_execution(
        self,
        query: str,
        params: QueryParams,
        result: Optional[QueryResult] = None,
        param_style: str = "inline",
    ) -> str:
        """
        Format complete query execution information for LLM analysis.

        Args:
            query: SQL query string
            params: Query parameters
            result: Query result (optional)
            param_style: Parameter formatting style ("inline", "table", "json")

        Returns:
            LLM-optimized formatted query execution display

        Example:
            >>> formatter = QueryDisplayFormatter()
            >>> display = formatter.format_query_execution(query, params, result)
        """
        sections = []

        # Business context header
        sections.append("-- QUERY EXECUTION CONTEXT")
        sections.append("")

        # Formatted query with context
        formatted_query = self.llm_formatter.format_query_for_llm(query, params)
        sections.append(formatted_query)

        # Execution results if available
        if result:
            sections.append("")
            sections.append("-- EXECUTION RESULTS")
            sections.append(f"-- Rows returned: {result.row_count}")
            sections.append(
                f"-- Execution time: {result.execution_time_seconds:.2f} seconds"
            )

            if result.metadata:
                if result.metadata.get("bytes_processed"):
                    sections.append(
                        f"-- Bytes processed: {result.metadata['bytes_processed']:,}"
                    )
                if result.metadata.get("cache_hit"):
                    sections.append(
                        f"-- Cache hit: {'Yes' if result.metadata['cache_hit'] else 'No'}"
                    )

            # Sample data preview for LLM
            if result.rows and len(result.rows) > 0:
                sections.append("")
                sections.append("-- SAMPLE RESULTS (first 3 rows)")
                sample_rows = result.rows[:3]
                for i, row in enumerate(sample_rows, 1):
                    if isinstance(row, dict):
                        row_data = ", ".join([f"{k}: {v}" for k, v in row.items()])
                        sections.append(f"-- Row {i}: {row_data}")

        return "\n".join(sections)

    def format_for_logging(
        self, query: str, params: QueryParams, result: Optional[QueryResult] = None
    ) -> str:
        """
        Format query information for logging (compact format).

        Args:
            query: SQL query string
            params: Query parameters
            result: Query result (optional)

        Returns:
            Compact formatted string for logging
        """
        # Compact one-line format for logging
        query_preview = query.strip().replace("\n", " ")[:100]
        if len(query.strip()) > 100:
            query_preview += "..."

        param_summary = self.param_formatter.format_parameters(params, "inline")

        log_parts = [f"Query: {query_preview}"]

        if params:
            log_parts.append(f"Params: {param_summary}")

        if result:
            log_parts.append(
                f"Result: {result.row_count} rows in {result.execution_time_seconds:.2f}s"
            )

        return " | ".join(log_parts)


# Convenience functions
def format_sql_query(query: str, indent_size: int = 2) -> str:
    """Convenience function to format SQL query for LLM analysis."""
    formatter = SQLQueryFormatter(indent_size=indent_size)
    return formatter.format_sql(query)


def format_query_for_llm_analysis(
    query: str,
    params: QueryParams,
    business_context: Optional[str] = None,
    analysis_type: str = "explain",
) -> str:
    """Convenience function to format query for LLM analysis."""
    formatter = LLMQueryFormatter()
    if analysis_type in ["explain", "optimize", "validate"]:
        return formatter.create_query_analysis_prompt(query, params, analysis_type)
    else:
        return formatter.format_query_for_llm(query, params, business_context)


def format_query_params(params: QueryParams, style: str = "inline") -> str:
    """Convenience function to format query parameters (optimized for LLM)."""
    formatter = ParameterFormatter()
    return formatter.format_parameters(params, style)


def format_query_result(result: QueryResult, format_type: str = "summary") -> str:
    """Convenience function to format query result (optimized for LLM)."""
    formatter = ResultFormatter()
    return formatter.format_result(result, format_type)


def display_query_execution(
    query: str, params: QueryParams, result: Optional[QueryResult] = None
) -> None:
    """Convenience function to display complete query execution for LLM analysis."""
    formatter = QueryDisplayFormatter()
    display_text = formatter.format_query_execution(query, params, result)
    print(display_text)


# Example usage
if __name__ == "__main__":
    # Example SQL query
    sample_query = """
    SELECT fact_profit.fiscal_week, CASE WHEN fact_profit.calendar_day BETWEEN DATE_SUB(DATE_TRUNC(DATE_SUB(CURRENT_DATE('Australia/Sydney'), INTERVAL 92 DAY), WEEK), INTERVAL -1 DAY) AND DATE_TRUNC(DATE_SUB(CURRENT_DATE('Australia/Sydney'), INTERVAL 1 DAY), WEEK) THEN 'TY' ELSE 'LY' END AS fiscal_year FROM gcp-wow-ent-im-tbl-prod.gs_allgrp_fin_data.fin_group_profit_v AS fact_profit WHERE fact_profit.salesorg = @salesorg AND fact_profit.region IN UNNEST(@regions)
    """

    sample_params = {
        "salesorg": "1005",
        "regions": ["NSW", "VIC", "QLD"],
        "comparison_days_1": 92,
    }

    # Demonstrate LLM-optimized formatters
    print("=== LLM-OPTIMIZED SQL FORMATTING ===")
    print(format_sql_query(sample_query))

    print("\n=== LLM QUERY ANALYSIS FORMAT ===")
    llm_analysis = format_query_for_llm_analysis(
        sample_query,
        sample_params,
        business_context="Weekly store performance comparison between This Year and Last Year",
        analysis_type="explain",
    )
    print(llm_analysis)

    print("\n=== PARAMETER FORMATTING (INLINE) ===")
    print(format_query_params(sample_params, "inline"))

    print("\n=== PARAMETER FORMATTING (TABLE) ===")
    print(format_query_params(sample_params, "table"))

    # Example of LLM query generation context
    print("\n=== LLM QUERY GENERATION CONTEXT ===")
    llm_formatter = LLMQueryFormatter()
    generation_context = llm_formatter.create_query_generation_context(
        business_requirement="Compare store sales performance across regions with competitor analysis",
        available_tables={
            "fin_group_profit_v": "Financial performance data by store and time period with competitor distance metrics",
            "dim_time_hierarchy_v": "Time dimension with fiscal periods and calendar mappings",
        },
        key_columns={
            "fiscal_week": "Fiscal week identifier in YYYY-WW format",
            "sales_excltax": "Sales revenue excluding tax in AUD",
            "salesorg": "Sales organization identifier (1005=Woolworths Supermarkets)",
            "region": "Geographic region code",
            "coles_competitor_distance": "Distance category to nearest Coles competitor",
        },
        sample_parameters={
            "salesorg": "1005",
            "regions": ["NSW", "VIC", "QLD"],
            "comparison_days_1": 92,
        },
    )
    print(generation_context)

    # Example of business logic annotations
    print("\n=== BUSINESS LOGIC ANNOTATIONS ===")
    annotations = {
        r"CASE.*TY.*LY": "This Year vs Last Year comparison logic",
        r"@salesorg": "Sales organization parameter for data filtering",
        r"coles_competitor_distance": "Competitive analysis by proximity to Coles stores",
        r"DATE_SUB.*INTERVAL.*DAY": "Dynamic date range calculation for period comparisons",
    }

    annotated_query = llm_formatter.annotate_query_with_business_logic(
        sample_query, annotations
    )
    print(annotated_query)
