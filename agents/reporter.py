import logging
import statistics
import os
import json
from typing import List, Dict, Any, Optional
from agents.state import AgentState
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure env vars are loaded from root
from dotenv import load_dotenv
import sys

# Calculate root path (assuming agents/ is one level deep)
root_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
# Force reload environment variables to pick up changes in .env
load_dotenv(root_path, override=True)

class ReporterAgent:
    """
    Analyzes the similar historical events to predict market impact and generate a confidence score.
    Uses LLM fallback if no history is found.
    """
    def __init__(self):
        # Explicitly get key to avoid 401
        api_key = os.getenv("OPENAI_API_KEY")
        
        # Debugging: Log if key is found (Masked)
        if api_key:
            logger.info(f"ReporterAgent loaded API Key: {api_key[:10]}...{api_key[-5:]}")
        else:
            logger.error("OPENAI_API_KEY is completely MISSING from environment!")
        
        # User confirmed 'gpt-5-nano' works with this key.
        self.llm = ChatOpenAI(model="gpt-5-nano", temperature=0, api_key=api_key)

    def run(self, state: AgentState) -> AgentState:
        """
        Executes the reporting step.
        """
        similar_events = state.get("similar_events", [])
        news_item = state.get("news_item", {})
        headline = news_item.get("headline", "")
        
        # --- Primary Logic: Historical Ground Truth ---
        if similar_events:
            logger.info(f"Analyzing {len(similar_events)} historical events...")

            # Initialize lists to collect returns
            day_1_returns = []
            day_7_returns = []
            day_30_returns = []
            similarity_scores = []

            for event in similar_events:
                d1 = event.get("day_1_return")
                d7 = event.get("day_7_return")
                d30 = event.get("day_30_return")
                score = event.get("similarity_score")
                
                if d1 is not None: day_1_returns.append(float(d1))
                if d7 is not None: day_7_returns.append(float(d7))
                if d30 is not None: day_30_returns.append(float(d30))
                if score is not None: similarity_scores.append(float(score))

            # Calculate averages
            avg_day_1 = statistics.mean(day_1_returns) if day_1_returns else 0.0
            avg_day_7 = statistics.mean(day_7_returns) if day_7_returns else 0.0
            avg_day_30 = statistics.mean(day_30_returns) if day_30_returns else 0.0
            avg_similarity = statistics.mean(similarity_scores) if similarity_scores else 0.0

            # Confidence Score Logic
            count_factor = min(len(similar_events), 5) / 5.0
            confidence_score = avg_similarity * count_factor
            
            analysis = {
                "average_returns": {
                    "day_1": round(avg_day_1, 2),
                    "day_7": round(avg_day_7, 2),
                    "day_30": round(avg_day_30, 2)
                },
                "confidence_score": round(confidence_score, 2),
                "sample_size": len(similar_events),
                "summary": f"Based on {len(similar_events)} similar historical events (Avg similarity: {avg_similarity:.2f}), the market showed an average 7-day return of {round(avg_day_7, 2)}%.",
                "source": "Historical Data"
            }
            
            logger.info(f"Analysis complete (History): {analysis['summary']}")
            state["analysis"] = analysis
            return state

        # --- Fallback Logic: LLM Prediction ---
        else:
            logger.warning("No similar events found. Falling back to LLM prediction.")
            try:
                prompt = ChatPromptTemplate.from_template(
                    """
                    You are a financial analyst.
                    Predict the short-term (1 day), medium-term (7 days), and long-term (30 days) market impact of the following news headline.
                    
                    Headline: {headline}
                    
                    Provide the estimated percentage returns and a confidence score (0.0 to 1.0).
                    Be conservative and realistic. If the news is neutral, returns should be near 0.
                    
                    Return ONLY a JSON object with this format:
                    {{
                        "average_returns": {{
                            "day_1": float,
                            "day_7": float,
                            "day_30": float
                        }},
                        "confidence_score": float,
                        "summary": "string explanation"
                    }}
                    """
                )
                
                chain = prompt | self.llm | JsonOutputParser()
                result = chain.invoke({"headline": headline})
                
                # Add source flag
                result["source"] = "AI Prediction (No History Found)"
                result["sample_size"] = 0
                
                logger.info(f"Analysis complete (LLM): {result.get('summary')}")
                state["analysis"] = result
                
            except Exception as e:
                logger.error(f"LLM Prediction failed with primary model: {e}")
                
                # Check if it was a model error (e.g., gpt-5-nano not found) and retry
                if "model_not_found" in str(e) or "does not exist" in str(e) or "404" in str(e):
                    logger.info("Retrying with fallback model: gpt-4o-mini")
                    try:
                        fallback_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=self.llm.openai_api_key)
                        chain = prompt | fallback_llm | JsonOutputParser()
                        result = chain.invoke({"headline": headline})
                        
                        result["source"] = "AI Prediction (Fallback Model)"
                        result["sample_size"] = 0
                        state["analysis"] = result
                        return state
                    except Exception as e2:
                        logger.error(f"Fallback LLM failed: {e2}")

                state["analysis"] = {
                    "summary": f"Unable to predict impact. Error: {str(e)}",
                    "confidence_score": 0.0,
                    "average_returns": {}
                }
                state["errors"] = state.get("errors", []) + [f"LLM Error: {str(e)}"]

            return state
