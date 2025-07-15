def get_trade_variance(
    department: str, week: int, year: int, versus_type: str
) -> float:
    """Pull trade component of variance (vs LY, Budget, or Forecast)."""
    # Base values by department and versus type
    base_values = {
        "Everyday Chilled": {"LY": 9.746, "Budget": -6.291, "Forecast": -1.040},
        "Bakery": {"LY": 7.2, "Budget": -4.8, "Forecast": -0.6},
        "default": {"LY": 5.2, "Budget": -3.5, "Forecast": -0.8},
    }

    # Get base value
    dept_key = department if department in ["Everyday Chilled", "Bakery"] else "default"
    base_value = base_values.get(dept_key, {}).get(versus_type, 0.0)

    # Apply week-specific adjustments
    if week == 51:  # Pre-Christmas
        if versus_type == "LY":
            # Strong performance vs last year
            multiplier = 1.8 if department == "Bakery" else 1.5
        elif versus_type == "Budget":
            # Better than budget due to strong sales
            multiplier = 0.6  # Less negative
        else:  # Forecast
            multiplier = 0.4  # Better than forecast
    elif week == 33:  # Back-to-school
        if versus_type == "LY":
            # Mixed performance
            multiplier = 0.7 if department == "Everyday Chilled" else 1.1
        elif versus_type == "Budget":
            # Slightly worse than budget
            multiplier = 1.2  # More negative
        else:  # Forecast
            multiplier = 1.5  # Worse than forecast
    else:
        multiplier = 1.0

    return round(base_value * multiplier, 3)


def get_mix_variance(department: str, week: int, year: int, versus_type: str) -> float:
    """Pull mix component of variance."""
    # Base values by department and versus type
    base_values = {
        "Everyday Chilled": {"LY": 0.168, "Budget": -1.375, "Forecast": 0.194},
        "Bakery": {"LY": 0.250, "Budget": -0.850, "Forecast": 0.150},
        "default": {"LY": 0.150, "Budget": -1.000, "Forecast": 0.200},
    }

    # Get base value
    dept_key = department if department in ["Everyday Chilled", "Bakery"] else "default"
    base_value = base_values.get(dept_key, {}).get(versus_type, 0.0)

    # Apply week-specific adjustments
    if week == 51:  # Pre-Christmas - premium mix
        if versus_type == "LY":
            adjustment = 0.5  # Add positive mix from premium
        elif versus_type == "Budget":
            adjustment = 0.8  # Less negative due to premium mix
        else:  # Forecast
            adjustment = 0.3  # Better mix than forecast
    elif week == 33:  # Back-to-school - value mix
        if versus_type == "LY":
            adjustment = -0.2  # Negative mix from value products
        elif versus_type == "Budget":
            adjustment = -0.3  # More negative due to value mix
        else:  # Forecast
            adjustment = -0.1  # Slightly worse mix
    else:
        adjustment = 0.0

    return round(base_value + adjustment, 3)


def get_claims_amount(department: str, week: int, year: int) -> float:
    """Pull vendor claims/rebates."""
    # Base values by department
    base_values = {"Everyday Chilled": 1.751, "Bakery": 0.650, "default": 0.850}

    base_value = base_values.get(department, base_values["default"])

    # Week-specific adjustments
    if week == 51:  # Pre-Christmas - suppliers invest heavily
        if department == "Everyday Chilled":
            return round(base_value * 2.8, 3)  # Major supplier support
        elif department == "Bakery":
            return round(base_value * 2.2, 3)  # Good supplier support
        else:
            return round(base_value * 2.0, 3)
    elif week == 33:  # Back-to-school
        if department == "Everyday Chilled":
            return round(base_value * 1.3, 3)  # Some promotional support
        elif department == "Bakery":
            return round(base_value * 1.5, 3)  # Lunch box promotions
        else:
            return round(base_value * 1.2, 3)
    else:
        return base_value


def get_rsa_amount(department: str, week: int, year: int) -> float:
    """Pull retail stock adjustment amounts."""
    # Base values by department
    base_values = {
        "Everyday Chilled": 0.555,
        "Bakery": 0.480,  # Higher than average due to wastage
        "default": 0.325,
    }

    base_value = base_values.get(department, base_values["default"])

    # Week-specific adjustments
    if week == 51:  # Pre-Christmas - higher stock movement
        if department == "Everyday Chilled":
            return round(base_value * 1.8, 3)  # High stock turnover
        elif department == "Bakery":
            return round(base_value * 2.5, 3)  # Very high wastage/adjustments
        else:
            return round(base_value * 1.6, 3)
    elif week == 33:  # Back-to-school
        if department == "Everyday Chilled":
            return round(base_value * 0.8, 3)  # Lower adjustments
        elif department == "Bakery":
            return round(base_value * 1.2, 3)  # Slightly higher for fresh items
        else:
            return round(base_value * 0.9, 3)
    else:
        return base_value
