def get_trade_variance(
    department: str, week: int, year: int, versus_type: str
) -> float:
    """Pull trade component of variance (vs LY, Budget, or Forecast)."""
    if versus_type == "LY":
        return 9.746 if department == "Everyday Chilled" else 5.2
    elif versus_type == "Budget":
        return -6.291 if department == "Everyday Chilled" else -3.5
    elif versus_type == "Forecast":
        return -1.040 if department == "Everyday Chilled" else -0.8
    else:
        return 0.0


def get_mix_variance(department: str, week: int, year: int, versus_type: str) -> float:
    """Pull mix component of variance."""
    if versus_type == "LY":
        return 0.168 if department == "Everyday Chilled" else 0.15
    elif versus_type == "Budget":
        return -1.375 if department == "Everyday Chilled" else -1.0
    elif versus_type == "Forecast":
        return 0.194 if department == "Everyday Chilled" else 0.2
    else:
        return 0.0


def get_claims_amount(department: str, week: int, year: int) -> float:
    """Pull vendor claims/rebates."""
    if department == "Everyday Chilled":
        return 1.751
    else:
        return 0.850


def get_rsa_amount(department: str, week: int, year: int) -> float:
    """Pull retail stock adjustment amounts."""
    if department == "Everyday Chilled":
        return 0.555
    else:
        return 0.325
