"""
Context Reduction Utilities for TradingAgents

This module provides utilities for managing LLM context windows by:
1. Counting tokens in text
2. Summarizing long texts to fit within token budgets
3. Truncating text when necessary
4. Managing token budgets across multiple context components

Based on context engineering best practices:
- https://www.philschmid.de/context-engineering
- https://www.philschmid.de/memory-in-agents
"""

import tiktoken
from typing import Dict, Optional, Tuple
from functools import lru_cache


# Default token budgets for different context components
DEFAULT_TOKEN_BUDGETS = {
    "market_report": 800,
    "sentiment_report": 600,
    "news_report": 800,
    "fundamentals_report": 800,
    "debate_history": 1000,
    "current_response": 500,
    "past_memories": 400,
    "system_prompt": 500,
    "total_max": 6000,  # Leave room for model response
}

# Summarization prompt template
SUMMARIZATION_PROMPT = """Summarize the following {report_type} report concisely while preserving:
1. Key numerical data and metrics
2. Main conclusions and recommendations
3. Critical risk factors or opportunities
4. Any actionable insights

Keep the summary focused and under {target_tokens} tokens.

Report to summarize:
{content}

Concise summary:"""


@lru_cache(maxsize=1)
def get_tokenizer(model: str = "gpt-4o-mini"):
    """
    Get a tokenizer for counting tokens. Cached for efficiency.
    
    Args:
        model: The model name to use for tokenization. 
               Currently supports OpenAI models via tiktoken.
    """
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base for unknown models
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Count the number of tokens in a text string.

    Args:
        text: The text to count tokens for
        model: The model name to use for tokenization

    Returns:
        Number of tokens in the text
    """
    if not text:
        return 0
    tokenizer = get_tokenizer(model)
    return len(tokenizer.encode(text))


def truncate_to_tokens(text: str, max_tokens: int, model: str = "gpt-4o-mini") -> str:
    """
    Truncate text to fit within a token limit.

    Args:
        text: The text to truncate
        max_tokens: Maximum number of tokens allowed
        model: The model name to use for tokenization

    Returns:
        Truncated text that fits within the token limit
    """
    if not text:
        return text

    tokenizer = get_tokenizer(model)
    tokens = tokenizer.encode(text)

    if len(tokens) <= max_tokens:
        return text

    # Truncate and decode
    truncated_tokens = tokens[:max_tokens]
    truncated_text = tokenizer.decode(truncated_tokens)

    # Add truncation indicator
    return truncated_text + "\n... [truncated due to length]"


def summarize_text(
    text: str,
    llm,
    report_type: str = "analysis",
    target_tokens: int = 500,
    model: str = "gpt-4o-mini",
    depth: int = 0,
    max_depth: int = 2
) -> str:
    """
    Use an LLM to summarize text to fit within a token budget.
    Supports recursive summarization if the output is still too large.

    Args:
        text: The text to summarize
        llm: The LLM to use for summarization
        report_type: Type of report (for prompt context)
        target_tokens: Target number of tokens for the summary
        model: The model name to use for tokenization
        depth: Current recursion depth
        max_depth: Maximum recursion depth

    Returns:
        Summarized text
    """
    current_tokens = count_tokens(text, model)

    # If already within budget, return as-is
    if current_tokens <= target_tokens:
        return text
        
    # Stop recursion if too deep
    if depth >= max_depth:
        return truncate_to_tokens(text, target_tokens, model)

    prompt = SUMMARIZATION_PROMPT.format(
        report_type=report_type,
        target_tokens=target_tokens,
        content=text
    )

    try:
        response = llm.invoke(prompt)
        summary = response.content if hasattr(response, 'content') else str(response)

        # Verify the summary fits
        if count_tokens(summary, model) > target_tokens:
            # Recursively summarize if still too long
            return summarize_text(
                summary, llm, report_type, target_tokens, model, depth + 1, max_depth
            )

        return summary
    except Exception as e:
        # Fallback to truncation if summarization fails
        print(f"Summarization failed: {e}. Falling back to truncation.")
        return truncate_to_tokens(text, target_tokens, model)


def summarize_debate_history(
    history: str,
    llm,
    max_tokens: int = 1000,
    model: str = "gpt-4o-mini"
) -> str:
    """
    Summarize debate history, keeping recent exchanges intact.

    Strategy: Keep the last exchange verbatim, summarize earlier ones.

    Args:
        history: The full debate history
        llm: The LLM to use for summarization
        max_tokens: Maximum tokens for the history
        model: The model name to use for tokenization

    Returns:
        Condensed debate history
    """
    if not history:
        return history

    current_tokens = count_tokens(history, model)
    if current_tokens <= max_tokens:
        return history

    # Split by analyst markers to identify exchanges
    lines = history.split('\n')
    exchanges = []
    current_exchange = []

    for line in lines:
        if line.startswith(('Bull Analyst:', 'Bear Analyst:')):
            if current_exchange:
                exchanges.append('\n'.join(current_exchange))
            current_exchange = [line]
        else:
            current_exchange.append(line)

    if current_exchange:
        exchanges.append('\n'.join(current_exchange))

    # Keep at least the last 2 exchanges intact
    if len(exchanges) <= 2:
        return truncate_to_tokens(history, max_tokens, model)

    # Summarize earlier exchanges, keep recent ones
    earlier_exchanges = '\n\n'.join(exchanges[:-2])
    recent_exchanges = '\n\n'.join(exchanges[-2:])

    recent_tokens = count_tokens(recent_exchanges, model)
    available_for_summary = max_tokens - recent_tokens - 50  # Buffer for labels

    if available_for_summary < 100:
        # Not enough room for summary, just keep recent
        return recent_exchanges

    summary_prompt = f"""Summarize the key points from this debate exchange concisely:

{earlier_exchanges}

Summary of earlier debate points:"""

    try:
        response = llm.invoke(summary_prompt)
        summary = response.content if hasattr(response, 'content') else str(response)
        summary = truncate_to_tokens(summary, available_for_summary, model)

        return f"[Earlier debate summary]: {summary}\n\n[Recent exchanges]:\n{recent_exchanges}"
    except Exception:
        return truncate_to_tokens(history, max_tokens, model)


def reduce_researcher_context(
    market_report: str,
    sentiment_report: str,
    news_report: str,
    fundamentals_report: str,
    history: str,
    current_response: str,
    past_memories: str,
    llm,
    token_budgets: Optional[Dict[str, int]] = None,
    model: str = "gpt-4o-mini"
) -> Dict[str, str]:
    """
    Reduce context for researcher agents to fit within token limits.

    This function applies summarization and truncation to ensure the total
    context fits within the model's context window. It uses a granular
    reduction strategy, targeting components that exceed their individual
    budgets first.

    Args:
        market_report: Market analysis report
        sentiment_report: Social media sentiment report
        news_report: News analysis report
        fundamentals_report: Company fundamentals report
        history: Debate history
        current_response: Current response from other analyst
        past_memories: Retrieved past memories
        llm: LLM to use for summarization
        token_budgets: Optional custom token budgets
        model: The model name to use for tokenization

    Returns:
        Dictionary containing the reduced context components
    """
    budgets = token_budgets or DEFAULT_TOKEN_BUDGETS
    
    # Map component names to their content and report types (for summarization)
    components = {
        "market_report": {"content": market_report, "type": "market analysis"},
        "sentiment_report": {"content": sentiment_report, "type": "social media sentiment"},
        "news_report": {"content": news_report, "type": "news analysis"},
        "fundamentals_report": {"content": fundamentals_report, "type": "company fundamentals"},
        "debate_history": {"content": history, "type": "debate history"},
        "current_response": {"content": current_response, "type": "response"},
        "past_memories": {"content": past_memories, "type": "memories"},
    }

    # Calculate current token usage
    current_usage = {k: count_tokens(v["content"], model) for k, v in components.items()}
    total_tokens = sum(current_usage.values())
    total_budget = budgets.get("total_max", 6000)

    # If within total budget, return as-is
    if total_tokens <= total_budget:
        return {k: v["content"] for k, v in components.items()}

    # Strategy:
    # 1. Identify components that exceed their individual budgets
    # 2. Reduce those components first
    # 3. If still over total budget, reduce largest remaining components

    reduced_content = {k: v["content"] for k, v in components.items()}
    
    # 1. Reduce components exceeding individual budgets
    for key, usage in current_usage.items():
        budget = budgets.get(key, 800) # Default fallback
        if usage > budget:
            content = components[key]["content"]
            report_type = components[key]["type"]
            
            if key == "debate_history":
                reduced_content[key] = summarize_debate_history(content, llm, budget, model)
            elif key in ["current_response", "past_memories"]:
                reduced_content[key] = truncate_to_tokens(content, budget, model)
            else:
                reduced_content[key] = summarize_text(content, llm, report_type, budget, model)
            
            # Update usage after reduction
            current_usage[key] = count_tokens(reduced_content[key], model)

    # Recalculate total
    total_tokens = sum(current_usage.values())
    
    # 2. If still over total budget, reduce proportionally
    if total_tokens > total_budget:
        # Sort components by size (descending)
        sorted_components = sorted(current_usage.items(), key=lambda x: x[1], reverse=True)
        
        for key, usage in sorted_components:
            if total_tokens <= total_budget:
                break
                
            # Calculate how much we need to shave off
            excess = total_tokens - total_budget
            
            # Don't reduce a component to nothing, keep at least 50% of its budget or 100 tokens
            min_tokens = max(budgets.get(key, 0) // 2, 100)
            
            if usage > min_tokens:
                # Target reduction: try to remove excess, but respect min_tokens
                reduction_target = max(usage - excess, min_tokens)
                
                # Apply further truncation (summarization is too slow for this iterative fix)
                reduced_content[key] = truncate_to_tokens(reduced_content[key], reduction_target, model)
                
                # Update usage
                new_usage = count_tokens(reduced_content[key], model)
                total_tokens -= (usage - new_usage)
                current_usage[key] = new_usage

    return reduced_content


def get_context_stats(
    market_report: str,
    sentiment_report: str,
    news_report: str,
    fundamentals_report: str,
    history: str,
    current_response: str,
    past_memories: str,
    model: str = "gpt-4o-mini"
) -> Dict[str, int]:
    """
    Get token statistics for context components.

    Useful for debugging and monitoring context usage.

    Args:
        All context components
        model: The model name to use for tokenization

    Returns:
        Dictionary with token counts for each component and total
    """
    stats = {
        "market_report": count_tokens(market_report, model),
        "sentiment_report": count_tokens(sentiment_report, model),
        "news_report": count_tokens(news_report, model),
        "fundamentals_report": count_tokens(fundamentals_report, model),
        "debate_history": count_tokens(history, model),
        "current_response": count_tokens(current_response, model),
        "past_memories": count_tokens(past_memories, model),
    }
    stats["total"] = sum(stats.values())
    return stats


# Default token budgets for risk management context
DEFAULT_RISK_TOKEN_BUDGETS = {
    "market_report": 600,
    "sentiment_report": 400,
    "news_report": 600,
    "fundamentals_report": 600,
    "risk_debate_history": 1200,
    "trader_plan": 400,
    "past_memories": 300,
    "other_responses": 600,  # Combined risky/safe/neutral responses
    "system_prompt": 400,
    "total_max": 5500,  # Leave room for model response (8192 - response tokens)
}


def summarize_risk_debate_history(
    history: str,
    llm,
    max_tokens: int = 1200,
    model: str = "gpt-4o-mini"
) -> str:
    """
    Summarize risk debate history, keeping recent exchanges intact.

    Strategy: Keep the last exchange from each analyst verbatim, summarize earlier ones.

    Args:
        history: The full risk debate history
        llm: The LLM to use for summarization
        max_tokens: Maximum tokens for the history
        model: The model name to use for tokenization

    Returns:
        Condensed risk debate history
    """
    if not history:
        return history

    current_tokens = count_tokens(history, model)
    if current_tokens <= max_tokens:
        return history

    # Split by analyst markers to identify exchanges
    lines = history.split('\n')
    exchanges = []
    current_exchange = []

    for line in lines:
        if line.startswith(('Risky Analyst:', 'Safe Analyst:', 'Neutral Analyst:')):
            if current_exchange:
                exchanges.append('\n'.join(current_exchange))
            current_exchange = [line]
        else:
            current_exchange.append(line)

    if current_exchange:
        exchanges.append('\n'.join(current_exchange))

    # Keep at least the last 3 exchanges intact (one from each analyst type)
    if len(exchanges) <= 3:
        return truncate_to_tokens(history, max_tokens, model)

    # Summarize earlier exchanges, keep recent ones
    earlier_exchanges = '\n\n'.join(exchanges[:-3])
    recent_exchanges = '\n\n'.join(exchanges[-3:])

    recent_tokens = count_tokens(recent_exchanges, model)
    available_for_summary = max_tokens - recent_tokens - 50  # Buffer for labels

    if available_for_summary < 100:
        # Not enough room for summary, just keep recent
        return recent_exchanges

    summary_prompt = f"""Summarize the key risk assessment points from this debate exchange concisely:

{earlier_exchanges}

Summary of earlier risk debate points:"""

    try:
        response = llm.invoke(summary_prompt)
        summary = response.content if hasattr(response, 'content') else str(response)
        summary = truncate_to_tokens(summary, available_for_summary, model)

        return f"[Earlier risk debate summary]: {summary}\n\n[Recent exchanges]:\n{recent_exchanges}"
    except Exception:
        return truncate_to_tokens(history, max_tokens, model)


def reduce_risk_management_context(
    market_report: str,
    sentiment_report: str,
    news_report: str,
    fundamentals_report: str,
    history: str,
    trader_plan: str,
    past_memories: str,
    other_responses: str,
    llm,
    token_budgets: Optional[Dict[str, int]] = None,
    model: str = "gpt-4o-mini"
) -> Dict[str, str]:
    """
    Reduce context for risk management agents to fit within token limits.

    This function applies summarization and truncation to ensure the total
    context fits within the model's context window (8192 tokens for gpt-4o-mini).

    Args:
        market_report: Market analysis report
        sentiment_report: Social media sentiment report
        news_report: News analysis report
        fundamentals_report: Company fundamentals report
        history: Risk debate history
        trader_plan: The trader's investment plan
        past_memories: Retrieved past memories
        other_responses: Combined responses from other analysts
        llm: LLM to use for summarization
        token_budgets: Optional custom token budgets
        model: The model name to use for tokenization

    Returns:
        Dictionary containing the reduced context components
    """
    budgets = token_budgets or DEFAULT_RISK_TOKEN_BUDGETS

    # Map component names to their content and report types
    components = {
        "market_report": {"content": market_report, "type": "market analysis"},
        "sentiment_report": {"content": sentiment_report, "type": "social media sentiment"},
        "news_report": {"content": news_report, "type": "news analysis"},
        "fundamentals_report": {"content": fundamentals_report, "type": "company fundamentals"},
        "risk_debate_history": {"content": history, "type": "risk debate history"},
        "trader_plan": {"content": trader_plan, "type": "trader plan"},
        "past_memories": {"content": past_memories, "type": "memories"},
        "other_responses": {"content": other_responses, "type": "analyst responses"},
    }

    # Calculate current token usage
    current_usage = {k: count_tokens(v["content"], model) for k, v in components.items()}
    total_tokens = sum(current_usage.values())
    total_budget = budgets.get("total_max", 5500)

    # If within total budget, return as-is
    if total_tokens <= total_budget:
        return {k: v["content"] for k, v in components.items()}

    reduced_content = {k: v["content"] for k, v in components.items()}

    # 1. Reduce components exceeding individual budgets
    for key, usage in current_usage.items():
        budget = budgets.get(key, 600)  # Default fallback
        if usage > budget:
            content = components[key]["content"]
            report_type = components[key]["type"]

            if key == "risk_debate_history":
                reduced_content[key] = summarize_risk_debate_history(content, llm, budget, model)
            elif key in ["trader_plan", "past_memories", "other_responses"]:
                reduced_content[key] = truncate_to_tokens(content, budget, model)
            else:
                reduced_content[key] = summarize_text(content, llm, report_type, budget, model)

            # Update usage after reduction
            current_usage[key] = count_tokens(reduced_content[key], model)

    # Recalculate total
    total_tokens = sum(current_usage.values())

    # 2. If still over total budget, reduce proportionally
    if total_tokens > total_budget:
        # Sort components by size (descending)
        sorted_components = sorted(current_usage.items(), key=lambda x: x[1], reverse=True)

        for key, usage in sorted_components:
            if total_tokens <= total_budget:
                break

            # Calculate how much we need to shave off
            excess = total_tokens - total_budget

            # Don't reduce a component to nothing, keep at least 50% of its budget or 100 tokens
            min_tokens = max(budgets.get(key, 0) // 2, 100)

            if usage > min_tokens:
                # Target reduction: try to remove excess, but respect min_tokens
                reduction_target = max(usage - excess, min_tokens)

                # Apply further truncation
                reduced_content[key] = truncate_to_tokens(reduced_content[key], reduction_target, model)

                # Update usage
                new_usage = count_tokens(reduced_content[key], model)
                total_tokens -= (usage - new_usage)
                current_usage[key] = new_usage

    return reduced_content
