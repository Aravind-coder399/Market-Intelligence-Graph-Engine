from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    """
    State definition for the Market Intelligence LangGraph.
    """
    news_item: Dict[str, str]  # Input: {headline, date, source}
    similar_events: List[Dict[str, Any]] # Output of Researcher: List of similar past events with returns
    analysis: Optional[Dict[str, Any]] # Output of Reporter: Analysis result, confidence score
    errors: Optional[List[str]] # To track any errors during processing
