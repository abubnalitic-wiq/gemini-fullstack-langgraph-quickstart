# Python Coding Guidelines

## Introduction

This document outlines coding standards and best practices for our Python projects. Following these guidelines ensures code consistency, maintainability, and quality across our repositories. Our projects typically use Python backends with Dash frontends, deployed on Google Cloud Platform.

## Environment Setup

### Package Management

- Use **uv** for package management
  - Enforce using only system Python with: 
  ```
  uv venv --python-preference only-system
  ```
  - Use uv for installing dependencies and managing virtual environments
  - Keep dependencies updated and minimize version conflicts

### Dependencies

- Maintain separate requirement files for development and production
- Pin dependency versions for production environments
- Document purpose of non-standard dependencies

## Code Formatting and Style

### Formatting Tools

- **Black**: We use Black as our standard code formatter
  - Maximum line length is 120 characters
  - Configure your IDE to run Black on save
  - For VSCode users, install recommended extensions for automatic formatting

### Naming Conventions

- Use `snake_case` for variables, functions, and modules
- Use `PascalCase` for classes
- Use `UPPER_CASE` for constants
- Choose descriptive names that indicate purpose

```python
# Good
user_profile = get_active_user(user_id)
class QueryProcessor:
    pass
MAX_RETRY_ATTEMPTS = 3

# Avoid
up = getactiveuser(uid)
class queryprocessor:
    pass
max_retry = 3
```

### Code Organization

- Keep functions focused and under 50 lines where possible
- Limit nesting to 3 levels or less
- Group related functionality within classes or modules
- Maintain a clean separation of concerns

## Documentation

### Docstrings

- All functions, classes, and modules should have docstrings
- Use clear descriptions of purpose, inputs, outputs, and exceptions
- Include usage examples for complex functions

```python
def process_query(query: str, schema_name: str = "finance") -> dict:
    """
    Process a natural language query against a specific data schema.
    
    Args:
        query: The natural language query to process
        schema_name: The schema to query against (default: "finance")
        
    Returns:
        Dictionary containing the processed query results
        
    Raises:
        SchemaError: If the schema is invalid or inaccessible
        QueryProcessingError: If the query cannot be processed
        
    Example:
        >>> result = process_query("Show me sales for last month")
    """
    # Implementation
```

### Comments

- Use comments to explain "why" not "what"
- Keep comments up-to-date with code changes
- Use TODO comments for planned improvements, with ticket references where applicable

## Development Best Practices

### Type Hints

- Use type hints consistently for all function parameters and return values
- Import types from `typing` module (List, Dict, Optional, Union, etc.)
- Use type aliases for complex types

```python
from typing import List, Dict, Optional, Union, TypeAlias

# Define type aliases for complex types
QueryResult: TypeAlias = Dict[str, Union[str, List[str], float]]

def get_query_results(
    query: str, 
    filters: Optional[Dict[str, str]] = None
) -> List[QueryResult]:
    """Get results for the specified query."""
    # Implementation
```

### Error Handling

- Use specific exception types rather than catching generic exceptions
- Provide context in error messages
- Log errors with appropriate level and context
- Fail gracefully with user-friendly messages

```python
try:
    result = process_query(query)
except SchemaError as e:
    logger.error(f"Invalid schema for query '{query}': {str(e)}")
    raise CustomError(f"The data schema is invalid: {str(e)}")
except Exception as e:
    logger.exception(f"Unexpected error processing query: {str(e)}")
    raise CustomError("An unexpected error occurred while processing your query")
```

### Logging

- **Use loguru for all logging** throughout the application
- Configure loguru with appropriate sinks and log rotation
- Use structured logging with context
- Include appropriate metadata (user IDs, request IDs, etc.)
- Use appropriate log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)

```python
from loguru import logger

# Configure loguru (typically done in app initialization)
logger.configure(
    handlers=[
        {"sink": sys.stdout, "level": "INFO"},
        {"sink": "logs/app.log", "rotation": "10 MB", "level": "DEBUG"},
    ]
)

# Context-rich logging
logger.info(f"Processing query: {query}", extra={"user_id": user_id, "request_id": request_id})

# Error logging with context
logger.error(f"Query processing failed: {error_message}", extra={"query": query})

# Performance logging
logger.debug(f"Query processed in {elapsed_time}ms")
```

### Testing

- Write unit tests for all new functionality
- Use pytest as the testing framework
- Aim for high test coverage (>80%)
- Include both positive and negative test cases

## Library and Framework Usage

### Leveraging Existing Libraries

- Use established libraries rather than writing custom code:
  - **Data Processing**: Use pandas or numpy
  - **AI/ML Functionality**: Use LangChain or similar frameworks
  - **Cloud Storage**: Use Google Cloud client libraries
  - **UI Components**: Use Dash and its component libraries

```python
# PREFERRED: Using existing libraries
import pandas as pd
df = pd.read_csv(file_path)
filtered_data = df[df['column'] > threshold]

# AVOID: Custom implementations
with open(file_path, 'r') as f:
    lines = f.readlines()
    # Custom parsing logic...
```

### GCP Integration

- Use official Google Cloud client libraries
- Follow GCP security best practices (service accounts, IAM, etc.)
- Structure deployments for scalability (containerization, CI/CD)

## Frontend Development (Dash)

### Component Organization

- Organize components logically by function or page
- Keep components focused and single-purpose
- Maintain a clear separation between UI and business logic

### Styling

- Use Tailwind's core utility classes for styling
- Maintain consistent styling patterns
- Minimize custom CSS

```python
# Component with Tailwind styling
html.Div(
    [
        html.H3("Query Results", className="text-lg font-semibold mb-2"),
        html.Div(results_content, id="results-content"),
    ],
    className="bg-white rounded-lg shadow p-4 mb-4"
)
```

### Callback Patterns

- Follow Dash callback patterns for interactivity
- Use meaningful IDs for components
- Organize callbacks near related components

```python
@app.callback(
    Output("results-content", "children"),
    Input("submit-button", "n_clicks"),
    State("query-input", "value")
)
def update_results(n_clicks, query):
    if n_clicks is None or not query:
        return ""
    return process_query_results(query)
```

## Project Structure

Maintain a consistent project structure across repositories:

```
.
├── README.md                # Project documentation
├── pyproject.toml           # Project configuration and dependencies
├── requirements/            # Dependency files
│   ├── dev.txt              # Development dependencies
│   └── prod.txt             # Production dependencies
├── src/                     # Main source code
│   ├── backend/             # Backend modules
│   │   ├── api/             # API endpoints
│   │   ├── core/            # Core business logic
│   │   ├── database/        # Database connections
│   │   └── utils/           # Utility functions
│   └── frontend/            # Dash frontend
│       ├── app.py           # Main Dash application
│       ├── assets/          # Static assets
│       ├── components/      # UI components
│       └── layouts/         # Page layouts
├── tests/                   # Test suite
│   ├── integration/         # Integration tests
│   └── unit/                # Unit tests
└── scripts/                 # Utility scripts
```

## Conclusion

These guidelines aim to promote code quality, consistency, and maintainability across our projects. While we encourage adherence to these standards, we also recognize that they may evolve over time. When in doubt, favor readability and maintainability over strict adherence to guidelines.

For project-specific guidelines or exceptions, refer to the project's README or documentation.