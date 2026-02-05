"""
LLM Summary Generator for Market Intelligence
Uses OpenAI GPT or Claude for natural language analysis reports.
"""
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class LLMSummarizer:
    """Generate natural language analysis reports using LLMs."""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = None
        
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                logger.info("OpenAI client initialized")
            except ImportError:
                logger.warning("OpenAI package not installed")
        else:
            logger.warning("OPENAI_API_KEY not set - LLM features disabled")
    
    def generate_summary(self, prediction: Dict[str, Any], headline: str, ticker: str = None) -> str:
        """Generate a natural language summary of the prediction."""
        
        if not self.client:
            return self._fallback_summary(prediction, headline, ticker)
        
        try:
            evidence_text = ""
            for ev in prediction.get("evidence", [])[:3]:
                evidence_text += f"- {ev['headline']} (Day 1: {ev.get('d1_return', 'N/A')}%)\n"
            
            prompt = f"""You are a financial analyst AI. Analyze this news prediction and provide a brief, professional summary.

NEWS HEADLINE: "{headline}"
{f"COMPANY: {ticker}" if ticker else ""}

PREDICTION RESULTS:
- Sentiment: {prediction.get('sentiment', 'NEUTRAL')}
- AI Text Sentiment Score: {prediction.get('ai_sentiment', 0):.2f}
- Predicted Day 1 Return: {prediction.get('predicted_d1', 0)}%
- Predicted Day 30 Return: {prediction.get('predicted_d30', 0)}%
- Confidence: {prediction.get('confidence', 'LOW')}

HISTORICAL EVIDENCE:
{evidence_text if evidence_text else "No similar historical events found."}

Provide a 2-3 sentence professional summary explaining the prediction, key risks, and recommended action (BUY/HOLD/SELL/WATCH). Be concise and actionable."""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            return self._fallback_summary(prediction, headline, ticker)
    
    def _fallback_summary(self, prediction: Dict[str, Any], headline: str, ticker: str = None) -> str:
        """Generate a rule-based summary when LLM is unavailable."""
        sentiment = prediction.get("sentiment", "NEUTRAL")
        d1 = prediction.get("predicted_d1", 0)
        confidence = prediction.get("confidence", "LOW")
        evidence_count = len(prediction.get("evidence", []))
        
        # Determine action
        if sentiment == "POSITIVE" and d1 > 1:
            action = "CONSIDER BUYING"
        elif sentiment == "NEGATIVE" and d1 < -1:
            action = "CONSIDER SELLING"
        else:
            action = "HOLD/WATCH"
        
        summary = f"Based on {evidence_count} similar historical events, the market is likely to react "
        summary += f"{'positively' if sentiment == 'POSITIVE' else 'negatively' if sentiment == 'NEGATIVE' else 'neutrally'}. "
        summary += f"Expected Day 1 return: {d1}%. "
        summary += f"Confidence: {confidence}. "
        summary += f"Suggested Action: {action}."
        
        return summary


def compare_headlines(predictor, headlines: list, tickers: list = None) -> Dict[str, Any]:
    """Compare predictions for multiple headlines side-by-side."""
    
    results = []
    
    for i, headline in enumerate(headlines):
        ticker = tickers[i] if tickers and i < len(tickers) else None
        prediction = predictor.predict(headline, ticker=ticker)
        
        results.append({
            "headline": headline,
            "ticker": ticker,
            "sentiment": prediction.get("sentiment"),
            "confidence": prediction.get("confidence"),
            "ai_sentiment": prediction.get("ai_sentiment"),
            "predicted_d1": prediction.get("predicted_d1"),
            "predicted_d7": prediction.get("predicted_d7"),
            "predicted_d30": prediction.get("predicted_d30"),
            "evidence_count": len(prediction.get("evidence", []))
        })
    
    return {
        "comparisons": results,
        "summary": _generate_comparison_summary(results)
    }


def _generate_comparison_summary(results: list) -> str:
    """Generate a summary comparing multiple predictions."""
    if not results:
        return "No results to compare."
    
    # Find best and worst
    best = max(results, key=lambda x: x.get("predicted_d1", 0))
    worst = min(results, key=lambda x: x.get("predicted_d1", 0))
    
    summary = f"📊 **Comparison Summary**\n\n"
    summary += f"**Most Bullish**: {best['headline'][:50]}... ({best['predicted_d1']}% D1)\n"
    summary += f"**Most Bearish**: {worst['headline'][:50]}... ({worst['predicted_d1']}% D1)\n"
    
    return summary
