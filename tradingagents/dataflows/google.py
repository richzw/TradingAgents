from typing import Annotated
from .googlenews_utils import getNewsData


def get_google_news(
    ticker: Annotated[str, "Ticker symbol or query to search with"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """
    Retrieve Google News for a given ticker/query within a date range.

    Args:
        ticker: Ticker symbol or query to search with
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format

    Returns:
        str: Formatted string containing news articles
    """
    query = ticker.replace(" ", "+")

    news_results = getNewsData(query, start_date, end_date)

    news_str = ""

    for news in news_results:
        news_str += (
            f"### {news['title']} (source: {news['source']}) \n\n{news['snippet']}\n\n"
        )

    if len(news_results) == 0:
        return ""

    return f"## {query} Google News, from {start_date} to {end_date}:\n\n{news_str}"