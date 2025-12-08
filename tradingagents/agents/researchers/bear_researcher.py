from langchain_core.messages import AIMessage
import time
import json

from tradingagents.agents.utils.context_reduction import (
    reduce_researcher_context,
    get_context_stats,
)


def create_bear_researcher(llm, memory, summarization_llm=None, config=None):
    """
    Create a bear researcher node.

    Args:
        llm: The LLM to use for generating arguments
        memory: The memory store for retrieving past experiences
        summarization_llm: Optional LLM for context summarization (defaults to llm)
        config: Optional config dict with token_budgets and enable_context_reduction
    """
    # Use the main LLM for summarization if not provided
    summary_llm = summarization_llm or llm

    # Get config settings
    enable_reduction = config.get("enable_context_reduction", True) if config else True
    token_budgets = config.get("token_budgets", None) if config else None

    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        # Apply context reduction if enabled
        if enable_reduction:
            reduced_context = reduce_researcher_context(
                market_research_report,
                sentiment_report,
                news_report,
                fundamentals_report,
                history,
                current_response,
                past_memory_str,
                summary_llm,
                token_budgets,
            )
            
            market_research_report = reduced_context["market_report"]
            sentiment_report = reduced_context["sentiment_report"]
            news_report = reduced_context["news_report"]
            fundamentals_report = reduced_context["fundamentals_report"]
            history = reduced_context["debate_history"]
            current_response = reduced_context["current_response"]
            past_memory_str = reduced_context["past_memories"]

        prompt = f"""You are a Bear Analyst making the case against investing in the stock. Your goal is to present a well-reasoned argument emphasizing risks, challenges, and negative indicators. Leverage the provided research and data to highlight potential downsides and counter bullish arguments effectively.

Key points to focus on:

- Risks and Challenges: Highlight factors like market saturation, financial instability, or macroeconomic threats that could hinder the stock's performance.
- Competitive Weaknesses: Emphasize vulnerabilities such as weaker market positioning, declining innovation, or threats from competitors.
- Negative Indicators: Use evidence from financial data, market trends, or recent adverse news to support your position.
- Bull Counterpoints: Critically analyze the bull argument with specific data and sound reasoning, exposing weaknesses or over-optimistic assumptions.
- Engagement: Present your argument in a conversational style, directly engaging with the bull analyst's points and debating effectively rather than simply listing facts.

Resources available:

Market research report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
Company fundamentals report: {fundamentals_report}
Conversation history of the debate: {history}
Last bull argument: {current_response}
Reflections from similar situations and lessons learned: {past_memory_str}
Use this information to deliver a compelling bear argument, refute the bull's claims, and engage in a dynamic debate that demonstrates the risks and weaknesses of investing in the stock. You must also address reflections and learn from lessons and mistakes you made in the past.
"""

        response = llm.invoke(prompt)

        argument = f"Bear Analyst: {response.content}"

        # Use original history from state for storing (not reduced)
        original_history = investment_debate_state.get("history", "")
        original_bear_history = investment_debate_state.get("bear_history", "")

        new_investment_debate_state = {
            "history": original_history + "\n" + argument,
            "bear_history": original_bear_history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bear_node
