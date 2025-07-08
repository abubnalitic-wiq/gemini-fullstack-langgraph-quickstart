# Financial Insights Project Onboarding Guide

Welcome to the Financial Insights project! This guide will help you understand the codebase, architecture, and development workflow.

## Project Overview

This project provides a Python-based analytics system for financial data analysis. It uses a combination of specialized query tools and an agent-based architecture to analyze financial metrics across different departments, compare performance against budgets and forecasts, and examine promotional effectiveness.

### Key Features

- Weekly retail metrics analysis by department
- Promotional sales and effectiveness tracking
- Variance analysis against budget, forecast, and prior year
- Time series analysis of key metrics
- Product category and channel performance breakdowns
- External impact event tracking

## Repository Structure

The project follows a structured organization:

```
.
├── build/                  # Build artifacts
├── src/                    # Source code
│   ├── query_tools/        # Core data query functionality
│   │   ├── aggregations.py        # MTD/YTD calculations
│   │   ├── budget_forecast_data.py # Budget/forecast comparisons
│   │   ├── events.py              # External event impacts
│   │   ├── financial_data_tools.py # Langchain tool wrappers
│   │   ├── metric_retrieval.py    # Core metric fetching
│   │   ├── product_hierarchy.py   # Category-level metrics
│   │   ├── promo.py               # Promotional metrics
│   │   ├── time_series.py         # Time-based analysis
│   │   └── variances.py           # Variance component analysis
│   ├── research_agent/     # Agent-based research functionality
│   │   ├── app.py                 # Main agent application
│   │   ├── configuration.py       # Agent configuration
│   │   ├── graph.py               # Agent workflow graph
│   │   ├── prompts.py             # Agent prompt templates
│   │   ├── state.py               # Agent state management
│   │   ├── tools_and_schemas.py   # Tool definitions
│   │   └── utils.py               # Utility functions
├── tests/                  # Test suite
│   ├── integration_tests/  # Integration tests
│   └── unit_tests/         # Unit tests
├── pyproject.toml          # Project configuration
└── uv.lock                 # Dependency lock file
```

## Core Concepts

### Query Tools

The `query_tools` package contains modular components for fetching and analyzing different types of retail metrics:

- **Metric Retrieval**: Base functions for getting weekly metrics by department
- **Time Series**: Functions for analyzing metrics over time periods
- **Promotional Analysis**: Tools for analyzing promotional effectiveness
- **Variance Analysis**: Functions for comparing actual vs. expected performance
- **Product Hierarchy**: Category-level breakdowns of performance

### Agent Architecture

The project uses LangChain for creating a research agent that can:

1. Parse natural language queries about retail performance
2. Select and execute appropriate query tools
3. Synthesize results into coherent analyses
4. Handle multi-step reasoning for complex analytics questions

## Development Environment Setup

1. **Clone the repository**

2. **Set up a virtual environment using uv**
   ```bash
   uv venv --python-preference only-system
   ```

3. **Activate the virtual environment**
   ```bash
   # For macOS/Linux
   source .venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   uv pip install -e .
   ```

## Key Workflows

### Adding a New Metric

1. Add the metric to the base metrics dictionary in `metric_retrieval.py`
2. Update relevant analysis functions to support the new metric
3. Update any dependent tools in `financial_data_tools.py`
4. Add tests for the new metric functionality

### Extending Department Coverage

To add support for a new retail department:

1. Add the department as a new condition in `get_weekly_metrics()` in `metric_retrieval.py`
2. Define appropriate base metrics for the department
3. Update department-specific logic in other modules as needed
4. Add tests for the new department

### Running Tests

```bash
python run_tests.py
```

## Best Practices

Follow these practices when contributing to the codebase:

1. **Type Annotations**: Use type hints for all function parameters and return values
2. **Docstrings**: Document all functions, classes, and modules with clear docstrings
3. **Error Handling**: Use specific exception types and provide context in error messages
4. **Logging**: Use loguru for logging with appropriate levels and context
5. **Testing**: Write unit tests for all new functionality with both positive and negative cases
6. **Code Formatting**: Use Black for code formatting with a maximum line length of 120 characters

## Common Tasks

### Fetching Weekly Metrics

```python
from src.query_tools.metric_retrieval import get_weekly_metrics

# Get sales and gross profit for Bakery department
metrics = get_weekly_metrics("Bakery", 33, 2024, ["sales", "gpbf"])
print(f"Sales: ${metrics['sales']}, GPBF: ${metrics['gpbf']}")
```

### Analyzing Promotional Performance

```python
from src.query_tools.promo import get_promo_sales

# Get promotional vs. non-promotional sales breakdown
promo_data = get_promo_sales("Everyday Chilled", 33, 2024)
print(f"Promo sales: ${promo_data['promo_sales']}")
print(f"Non-promo sales: ${promo_data['non_promo_sales']}")
```

### Running Variance Analysis

```python
from src.query_tools.financial_data_tools import fetch_variance_data

# Compare actual sales vs. budget, forecast, and last year
variance = fetch_variance_data("Everyday Chilled", 33, 2024, "sales")
print(f"Vs Budget: ${variance['vs_budget']}")
print(f"Vs Last Year: ${variance['vs_ly']}")
```

## Getting Help

If you need assistance, consult:
- The Python Coding Guidelines document for overall standards
- Code comments and docstrings for specific functionality