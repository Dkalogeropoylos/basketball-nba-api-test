from providers.nba_stats import NBAStatsProvider
from providers.balldontlie_wnba import BDLWNBA


def get_basic_provider(league: str):
    """
    Core/basic historical data:
      WNBA -> nba_api LeagueID 10
      NBA  -> nba_api LeagueID 00
    """
    league = league.upper()
    if league == "WNBA":
        return NBAStatsProvider("10"), "nba_api WNBA (basic)"
    if league == "NBA":
        return NBAStatsProvider("00"), "nba_api NBA (basic)"
    raise ValueError(f"{league} is not enabled yet.")


def get_advanced_provider(league: str, bdl_key: str | None = None):
    """
    Optional supplementary provider.
    BALLDONTLIE WNBA advanced/stat/injury endpoints depend on account tier.
    """
    league = league.upper()
    if league == "WNBA" and bdl_key:
        return BDLWNBA(bdl_key), "BALLDONTLIE WNBA (advanced/optional)"
    return None, None
