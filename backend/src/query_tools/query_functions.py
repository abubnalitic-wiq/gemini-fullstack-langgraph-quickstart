"""Retail Analytics Query Functions.

This module provides parameterized functions for generating SQL queries
for retail analytics and business intelligence reporting.

Uses BigQuery parameter syntax (@parameter) for secure query execution.
"""

from typing import Any, Dict, List, Optional, Tuple, TypeAlias

from loguru import logger

# Type aliases for better readability
SalesOrg: TypeAlias = str
Region: TypeAlias = str
FiscalYear: TypeAlias = str
TimeZone: TypeAlias = str
ComparisonTimeHierarchy: TypeAlias = str
MerchandiseManager: TypeAlias = str
QueryParams: TypeAlias = Dict[str, Any]
QueryResult: TypeAlias = Tuple[str, QueryParams]


class QueryParameterError(Exception):
    """Raised when query parameters are invalid."""

    pass


def weekly_store_performance_metrics(
    salesorg: SalesOrg,
    regions: List[Region],
    timezone_1: TimeZone,
    timezone_2: TimeZone,
    timezone_3: TimeZone,
    timezone_4: TimeZone,
    comparison_days_1: int,
    comparison_days_2: int,
    comparison_days_3: int,
    comparison_days_4: int,
    comparison_days_offset_1: int,
    comparison_days_offset_2: int,
) -> QueryResult:
    """Generate SQL query for weekly store performance metrics.

    Shows weekly sales, stock adjustments, stock loss, and customer counts by store.
    Limited by sales organization, region, and a comparison date range.

    Args:
        salesorg: Sales organization identifier
        regions: List of regions to include
        timezone_1: Timezone for first comparison period start
        timezone_2: Timezone for first comparison period end
        timezone_3: Timezone for second comparison period start
        timezone_4: Timezone for second comparison period end
        comparison_days_1: Days back from current date for first period start
        comparison_days_2: Days back from current date for first period end
        comparison_days_3: Days back from current date for second period start
        comparison_days_4: Days back from current date for second period end
        comparison_days_offset_1: Offset days for first period
        comparison_days_offset_2: Offset days for second period

    Returns:
        Tuple of (SQL query string, parameters dict) ready for BigQuery execution

    Raises:
        QueryParameterError: If parameters are invalid

    Example:
        >>> query, params = weekly_store_performance_metrics(
        ...     salesorg="1005",
        ...     regions=["NSW", "VIC", "QLD"],
        ...     timezone_1="Australia/Sydney",
        ...     timezone_2="Australia/Sydney",
        ...     timezone_3="Australia/Sydney",
        ...     timezone_4="Australia/Sydney",
        ...     comparison_days_1=92,
        ...     comparison_days_2=1,
        ...     comparison_days_3=456,
        ...     comparison_days_4=364,
        ...     comparison_days_offset_1=-1,
        ...     comparison_days_offset_2=-1
        ... )
    """
    # Parameter validation
    if not salesorg:
        raise QueryParameterError("salesorg parameter is required")

    if not regions:
        raise QueryParameterError("regions must be a non-empty list")

    for tz in [timezone_1, timezone_2, timezone_3, timezone_4]:
        if not tz:
            raise QueryParameterError("All timezone parameters are required")

    logger.debug(
        f"Generating weekly store performance query for salesorg: {salesorg}, regions: {regions}"
    )

    query = """
    SELECT
      fact_profit.fiscal_week,
      CASE
        WHEN fact_profit.calendar_day BETWEEN DATE_SUB(DATE_TRUNC(DATE_SUB(CURRENT_DATE('Australia/Sydney'), INTERVAL 92 DAY), WEEK), INTERVAL -1 DAY) AND DATE_TRUNC(DATE_SUB(CURRENT_DATE('Australia/Sydney'), INTERVAL 1 DAY), WEEK)
        THEN 'TY'
        WHEN fact_profit.calendar_day BETWEEN DATE_SUB(DATE_TRUNC(DATE_SUB(CURRENT_DATE('Australia/Sydney'), INTERVAL 456 DAY), WEEK), INTERVAL -1 DAY) AND DATE_TRUNC(DATE_SUB(CURRENT_DATE('Australia/Sydney'), INTERVAL 364 DAY), WEEK)
        THEN 'LY'
      END AS fiscal_year,
      fact_profit.site,
      fact_profit.site_description,
      CASE
        WHEN fact_profit.coles_competitor_distance = '<1 KM'
        THEN 'Less than 1km'
        WHEN fact_profit.coles_competitor_distance = '1-3 KM'
        THEN '1km to 3km'
        WHEN fact_profit.coles_competitor_distance = '3-5 KM'
        THEN '3km to 5km'
        WHEN fact_profit.coles_competitor_distance = '>5 KM'
        THEN 'Greater than 5km'
        WHEN fact_profit.coles_competitor_distance = 'Same Centre'
        THEN 'Same Centre'
      END AS coles_competitor_distance,
      fact_profit.store_segment,
      SUM(fact_profit.sales_excltax) AS sales,
      -SUM(fact_profit.total_stock_adjustments) + SUM(fact_profit.dumps_cost) AS stock_adj,
      SUM(fact_profit.total_stock_loss) AS tsl,
      Safe_Divide(
        -SUM(fact_profit.total_stock_adjustments) + SUM(fact_profit.dumps_cost),
        SUM(fact_profit.sales_excltax)
      ) AS stk_adj_rate,
      SUM(fact_profit.items_sold) AS itemssold,
      SUM(fact_profit.transaction_cnt_site) AS customers
    FROM gcp-wow-ent-im-tbl-prod.gs_allgrp_fin_data.fin_group_profit_v AS fact_profit
    WHERE
      fact_profit.salesorg = @salesorg
      AND fact_profit.region IN UNNEST(@regions)
      AND NOT fact_profit.zone LIKE '%CLSD'
      AND NOT fact_profit.zone LIKE '%REGX'
      AND (
        fact_profit.calendar_day BETWEEN DATE_SUB(DATE_TRUNC(DATE_SUB(CURRENT_DATE(@timezone_1), INTERVAL @comparison_days_1 DAY), WEEK), INTERVAL @comparison_days_offset_1 DAY) AND DATE_TRUNC(DATE_SUB(CURRENT_DATE(@timezone_2), INTERVAL @comparison_days_2 DAY), WEEK)
        OR fact_profit.calendar_day BETWEEN DATE_SUB(DATE_TRUNC(DATE_SUB(CURRENT_DATE(@timezone_3), INTERVAL @comparison_days_3 DAY), WEEK), INTERVAL @comparison_days_offset_2 DAY) AND DATE_TRUNC(DATE_SUB(CURRENT_DATE(@timezone_4), INTERVAL @comparison_days_4 DAY), WEEK)
      )
    GROUP BY ALL
    ORDER BY
      fiscal_week;
    """

    params = {
        "salesorg": salesorg,
        "regions": regions,
        "timezone_1": timezone_1,
        "timezone_2": timezone_2,
        "timezone_3": timezone_3,
        "timezone_4": timezone_4,
        "comparison_days_1": comparison_days_1,
        "comparison_days_2": comparison_days_2,
        "comparison_days_3": comparison_days_3,
        "comparison_days_4": comparison_days_4,
        "comparison_days_offset_1": comparison_days_offset_1,
        "comparison_days_offset_2": comparison_days_offset_2,
    }

    return query, params


def sales_vs_budget_by_fiscal_period(
    salesorg: SalesOrg, comparison_timehierarchy: ComparisonTimeHierarchy
) -> QueryResult:
    """# Generate SQL query for sales vs budget comparison by fiscal period.

    Shows total sales, operations budget, and merchandise budget by fiscal period.
    Limited to specified sales organization and comparison time hierarchy.

    Args:
        salesorg: Sales organization identifier
        comparison_timehierarchy: Time hierarchy for comparison period

    Returns:
        Tuple of (SQL query string, parameters dict) ready for BigQuery execution

    Raises:
        QueryParameterError: If parameters are invalid

    Example:
        >>> query, params = sales_vs_budget_by_fiscal_period(
        ...     salesorg="1005",
        ...     comparison_timehierarchy="FY2024Q4"
        ... )
    """
    if not salesorg:
        raise QueryParameterError("salesorg parameter is required")

    if not comparison_timehierarchy:
        raise QueryParameterError("comparison_timehierarchy parameter is required")

    logger.debug(
        f"Generating sales vs budget query for salesorg: {salesorg}, timehierarchy: {comparison_timehierarchy}"
    )

    query = """
    DECLARE last_comp_period_array ARRAY < DATE >;
    SET last_comp_period_array = (
      SELECT
        ARRAY_AGG(calday)
      FROM gcp-wow-ent-im-tbl-prod.adp_dm_masterdata_view.dim_time_hierarchy_v AS dim_date
      WHERE
        timehierarchy = @comparison_timehierarchy
    );
    SELECT
      fiscal_period,
      SUM(sales_excltax) AS actuals_sales,
      SUM(sales_ops_budget) AS ops_budget,
      SUM(sales_merch_budget) AS merch_budget
    FROM gcp-wow-ent-im-tbl-prod.gs_allgrp_fin_data.fin_group_profit_v AS fact_profit
    WHERE
      calendar_day IN UNNEST(last_comp_period_array) AND salesorg = @salesorg
    GROUP BY ALL;
    """

    params = {
        "salesorg": salesorg,
        "comparison_timehierarchy": comparison_timehierarchy,
    }

    return query, params


def weekly_sales_and_gross_profit_by_subcategory(
    fiscal_years: List[FiscalYear],
    salesorg: SalesOrg = "1005",
    merchandise_managers: Optional[List[MerchandiseManager]] = None,
) -> QueryResult:
    """Generate SQL query for weekly sales and gross profit by subcategory.

    Shows weekly sales, items sold, scanback, and gross profit metrics by subcategory,
    merchandise department, and state. Limited to sales organization '1005',
    specific merchandise managers, and parameterized fiscal years.

    Args:
        fiscal_years: List of fiscal years to include
        salesorg: Sales organization identifier (default: "1005")
        merchandise_managers: List of merchandise manager codes
                            (default: ["MM924_AU", "MM920_AU"])

    Returns:
        Tuple of (SQL query string, parameters dict) ready for BigQuery execution

    Raises:
        QueryParameterError: If parameters are invalid

    Example:
        >>> query, params = weekly_sales_and_gross_profit_by_subcategory(
        ...     fiscal_years=["2024", "2023"],
        ...     merchandise_managers=["MM924_AU", "MM920_AU"]
        ... )
    """
    if not fiscal_years:
        raise QueryParameterError("fiscal_years must be a non-empty list")

    if not salesorg:
        raise QueryParameterError("salesorg parameter is required")

    if merchandise_managers is None:
        merchandise_managers = ["MM924_AU", "MM920_AU"]

    if not merchandise_managers:
        raise QueryParameterError("merchandise_managers must be a non-empty list")

    logger.debug(
        f"Generating subcategory query for years: {fiscal_years}, managers: {merchandise_managers}"
    )

    query = """
    SELECT
      fact_profit.fiscal_year,
      fact_profit.fiscal_period,
      fact_profit.fiscal_week,
      fact_profit.subcategory,
      fact_profit.subcategory_description,
      fact_profit.merchandisemanager_department,
      fact_profit.state,
      sum(fact_profit.sales_excltax) AS sales,
      sum(fact_profit.items_sold) AS items,
      sum(fact_profit.scanback) AS scanback,
      sum(fact_profit.initial_gp) AS initial_gp,
      sum(fact_profit.total_stock_adjustments) AS total_stock_adjustments,
      sum(fact_profit.interim_gp) AS interim_gp
    FROM gcp-wow-ent-im-tbl-prod.gs_allgrp_fin_data.fin_group_profit_v AS fact_profit
    WHERE
      fact_profit.salesorg = @salesorg
      AND fact_profit.fiscal_year IN UNNEST(@fiscal_years)
      AND fact_profit.merchandisemanager_code IN UNNEST(@merchandise_managers)
    GROUP BY
      fact_profit.fiscal_year,
      fact_profit.fiscal_period,
      fact_profit.fiscal_week,
      fact_profit.subcategory,
      fact_profit.subcategory_description,
      fact_profit.merchandisemanager_department,
      fact_profit.state
    ORDER BY
      fiscal_year,
      fiscal_period,
      fiscal_week;
    """

    params = {
        "salesorg": salesorg,
        "fiscal_years": fiscal_years,
        "merchandise_managers": merchandise_managers,
    }

    return query, params


def store_sales_profitability_weekly_trend(
    salesorg: SalesOrg, fiscal_year: FiscalYear
) -> QueryResult:
    """Generate SQL query for store sales and profitability weekly trends.

    Shows weekly and monthly sales, units, transactions, and gross profit by store
    and sales channel. Limited to a specific sales organization and fiscal year.

    Args:
        salesorg: Sales organization identifier
        fiscal_year: Fiscal year for the analysis

    Returns:
        Tuple of (SQL query string, parameters dict) ready for BigQuery execution

    Raises:
        QueryParameterError: If parameters are invalid

    Example:
        >>> query, params = store_sales_profitability_weekly_trend(
        ...     salesorg="1005",
        ...     fiscal_year="2024"
        ... )
    """
    if not salesorg:
        raise QueryParameterError("salesorg parameter is required")

    if not fiscal_year:
        raise QueryParameterError("fiscal_year parameter is required")

    logger.debug(
        f"Generating store sales trend query for salesorg: {salesorg}, fiscal_year: {fiscal_year}"
    )

    query = """
    SELECT
      fact_profit.fiscal_year,
      fact_profit.fiscal_period,
      CAST(SUBSTR(fact_profit.fiscal_period, LENGTH(fact_profit.fiscal_period) - 1, 2) AS INT64) AS month,
      fact_profit.fiscal_week,
      CAST(SUBSTR(fact_profit.fiscal_week, LENGTH(fact_profit.fiscal_week) - 1, 2) AS INT64) AS week,
      CASE
        WHEN fact_profit.region_description = 'SM REG 01 STORE OPS - WA'
        THEN 'SANTWA'
        ELSE CASE
          WHEN fact_profit.region_description = 'SM REG 02 STORE OPS - SA / NT'
          THEN 'SANTWA'
          ELSE CASE
            WHEN fact_profit.region_description = 'SM REG 03 STORE OPS - VIC / TAS'
            THEN 'VIC/TAS'
            ELSE CASE
              WHEN fact_profit.region_description = 'SM REG 04 STORE OPS - NSW /ACT'
              THEN 'NSW/ACT'
              ELSE CASE
                WHEN fact_profit.region_description = 'SM REG 05 STORE OPS - QLD'
                THEN 'QLD'
                ELSE CASE
                  WHEN fact_profit.region_description = 'SM REG 99 STORE OPS - CFC'
                  THEN 'CFC'
                  ELSE 'Unmapped'
                END
              END
            END
          END
        END
      END AS financial_state,
      fact_profit.region_description,
      fact_profit.zone_description,
      fact_profit.site,
      fact_profit.site_description,
      CASE
        WHEN fact_profit.store_cluster LIKE '%BUDGET%'
        THEN 'Budget'
        ELSE CASE
          WHEN fact_profit.store_cluster LIKE '%MAINSTREAM%'
          THEN 'Mainstream'
          ELSE CASE
            WHEN fact_profit.store_cluster LIKE '%PREMIUM%'
            THEN 'Premium'
            ELSE 'Unmapped'
          END
        END
      END AS store_cluster_parent,
      fact_profit.store_cluster,
      fact_profit.coles_competitor_distance,
      fact_profit.aldi_competitor_distance,
      CASE
        WHEN fact_profit.sales_channel LIKE '%CC%'
        THEN 'Online'
        ELSE CASE WHEN fact_profit.sales_channel LIKE '%HD%' THEN 'Online' ELSE 'B&M' END
      END AS sales_channel_grandparent,
      CASE
        WHEN fact_profit.sales_channel LIKE '%CC%'
        THEN 'C&C'
        ELSE CASE WHEN fact_profit.sales_channel LIKE '%HD%' THEN 'HD' ELSE 'B&M' END
      END AS sales_channel_parent,
      fact_profit.sales_channel,
      ROUND(SUM(fact_profit.sales_excltax), 2) AS sales,
      ROUND(SUM(fact_profit.promo_sales), 2) AS sales_promotional,
      ROUND(SUM(fact_profit.ownbrand_sales), 2) AS sales_ownbrand,
      ROUND(SUM(fact_profit.items_sold), 2) AS items_sold,
      ROUND(SUM(fact_profit.transaction_cnt_site), 2) AS transactions,
      ROUND(SUM(fact_profit.initial_gp), 2) AS initial_gp,
      ROUND(SUM(fact_profit.cogs), 2) AS cogs
    FROM gcp-wow-ent-im-tbl-prod.gs_allgrp_fin_data.fin_group_profit_v AS fact_profit
    WHERE
      fact_profit.salesorg = @salesorg AND fact_profit.fiscal_year = @fiscal_year
    GROUP BY
      fact_profit.fiscal_year,
      fact_profit.fiscal_period,
      month,
      fact_profit.fiscal_week,
      week,
      financial_state,
      fact_profit.region_description,
      fact_profit.zone_description,
      fact_profit.site,
      fact_profit.site_description,
      store_cluster_parent,
      fact_profit.store_cluster,
      fact_profit.coles_competitor_distance,
      fact_profit.aldi_competitor_distance,
      sales_channel_grandparent,
      sales_channel_parent,
      fact_profit.sales_channel
    ORDER BY
      fiscal_year,
      fiscal_period,
      month,
      fiscal_week,
      week,
      financial_state,
      region_description,
      zone_description,
      site;
    """

    params = {
        "salesorg": salesorg,
        "fiscal_year": fiscal_year,
    }

    return query, params


# Example usage
def example_usage():
    """Demonstrate usage of the retail analytics query functions."""
    try:
        # Weekly store performance
        query, params = weekly_store_performance_metrics(
            salesorg="1005",
            regions=["1000SR01-SOPS", "1000SR02-SOPS"],
            timezone_1="Australia/Sydney",
            timezone_2="Australia/Sydney",
            timezone_3="Australia/Sydney",
            timezone_4="Australia/Sydney",
            comparison_days_1=92,
            comparison_days_2=1,
            comparison_days_3=456,
            comparison_days_4=364,
            comparison_days_offset_1=-1,
            comparison_days_offset_2=-1,
        )
        logger.info("Generated weekly store performance query with parameters")

        # Sales vs budget
        budget_query, budget_params = sales_vs_budget_by_fiscal_period(
            salesorg="1005", comparison_timehierarchy="Last Completed Period TY"
        )
        logger.info("Generated sales vs budget query with parameters")

        return {
            "weekly_performance": (query, params),
            "sales_vs_budget": (budget_query, budget_params),
        }

    except QueryParameterError as e:
        logger.error(f"Parameter validation error: {str(e)}")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error generating queries: {str(e)}")
        raise
