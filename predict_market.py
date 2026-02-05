import sys
import os
import argparse
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from textblob import TextBlob

# Ensure we can import graph client
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from graph.falkor_client import FalkorClient

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class MarketPredictor:
    def __init__(self):
        self.falkor = FalkorClient()
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def predict(self, headline, ticker=None, top_k=5):
        logger.info(f"\n--- Analyzing: '{headline}' ---")
        
        # 0. AI Sentiment Analysis
        blob = TextBlob(headline)
        sentiment_score = blob.sentiment.polarity # -1.0 to 1.0
        logger.info(f"AI Sentiment Score: {sentiment_score:.2f}")

        # 1. Generate Embedding
        embedding = self.encoder.encode(headline).tolist()
        
        try:
            # 2. Find Similar News (Try Server-Side Vector Search First)
            query = """
            CALL db.idx.vector.queryNodes('News', 'embedding', $k, vecf32($vec)) 
            YIELD node, score
            RETURN node.headline, node.date, node.ticker, node.day_1_return, node.day_7_return, node.day_30_return, score
            """
            
            params = {
                'vec': embedding,
                'k': top_k
            }
            
            results = self.falkor.query(query, params)
            
        except Exception as e:
            logger.warning(f"Server-Side Vector Search failed ({e}). Switching to Client-Side Search...")
            results = self._fallback_client_side_search(embedding, top_k)

        if not results:
            logger.warning("No similar historical events found in graph. Trying Finnhub fallback...")
            return self._finnhub_fallback(headline, ticker, sentiment_score)


        # 3. Analyze Results
        print(f"\nFound {len(results)} similar historical events:\n")
        
        day_1_returns = []
        day_7_returns = []
        
        for idx, row in enumerate(results):
            hist_headline = row[0]
            date = row[1]
            hist_ticker = row[2]
            d1 = row[3]
            d7 = row[4]
            score = row[5]
            
            # Validations for display
            d1_disp = f"{d1:.2f}%" if d1 is not None else "N/A"
            d7_disp = f"{d7:.2f}%" if d7 is not None else "N/A"

            print(f"{idx+1}. [{score:.2f}] {date} ({hist_ticker}): {hist_headline[:80]}...")
            print(f"    Impact: Day 1: {d1_disp}, Day 7: {d7_disp}")
            
            if d1 is not None: day_1_returns.append(d1)
            if d7 is not None: day_7_returns.append(d7)

        # 4. Synthesize Prediction
        prediction = {
            "sentiment": "NEUTRAL",
            "confidence": "LOW",
            "ai_sentiment": sentiment_score,
            "predicted_d1": 0.0,
            "predicted_d7": 0.0,
            "predicted_d30": 0.0,
            "evidence": []
        }

        # Format Evidence from Results
        day_1_returns = []
        day_7_returns = []
        day_30_returns = []

        for idx, row in enumerate(results):
            evidence = {
                "headline": row[0],
                "date": row[1],
                "ticker": row[2],
                "d1_return": row[3],
                "d7_return": row[4],
                "d30_return": row[5],
                "similarity_score": row[6]
            }
            prediction["evidence"].append(evidence)
            
            if row[3] is not None: day_1_returns.append(row[3])
            if row[4] is not None: day_7_returns.append(row[4])
            if row[5] is not None: day_30_returns.append(row[5])

        print("\n--- PREDICTION ---")
        if day_1_returns:
            avg_d1 = np.mean(day_1_returns)
            avg_d7 = np.mean(day_7_returns)
            avg_d30 = np.mean(day_30_returns) if day_30_returns else 0.0
            
            # Combine Historical + AI Signal
            # If both agree, confidence is higher
            hist_sentiment = 1 if avg_d1 > 0 else -1
            ai_signal = 1 if sentiment_score > 0.1 else (-1 if sentiment_score < -0.1 else 0)
            
            final_sentiment = "POSITIVE" if avg_d1 > 0 else "NEGATIVE"
            
            prediction.update({
                "sentiment": final_sentiment,
                "confidence": "HIGH" if (len(prediction["evidence"]) >= 3 and hist_sentiment == ai_signal) else "MEDIUM",
                "predicted_d1": round(avg_d1, 2),
                "predicted_d7": round(avg_d7, 2),
                "predicted_d30": round(avg_d30, 2),
                "summary": f"Based on {len(prediction['evidence'])} similar events, market reaction is likely {final_sentiment}."
            })
            
            print(f"Based on historical similarity, the market reaction is likely to be {final_sentiment}.")
            print(f"Expected Day 1 Return: {avg_d1:.2f}%")
            print(f"Expected Day 7 Return: {avg_d7:.2f}%")
        else:
            print("Insufficient return data to make a prediction.")
            prediction["summary"] = "Insufficient return data to make a prediction."
            
        return prediction

    def _fallback_client_side_search(self, target_embedding, k=5):
        """
        Fetches all news embeddings and computes cosine similarity locally.
        Useful when DB Vector Search is unavailable.
        """
        # Fetch all news with embeddings (support both Schema versions)
        query = """
        MATCH (n) 
        WHERE (n:News OR n:NewsEvent) AND n.embedding IS NOT NULL 
        RETURN n.headline, n.date, n.ticker, n.day_1_return, n.day_7_return, n.day_30_return, n.embedding
        """
        
        # Note: This could be heavy for very large graphs, but fine for <100k nodes
        rows = self.falkor.query(query)
        if not rows:
            return []
            
        scored_results = []
        target_vec = np.array(target_embedding)
        norm_target = np.linalg.norm(target_vec)
        
        for row in rows:
            # row format: [headline, date, ticker, d1, d7, d30, embedding]
            db_vec = np.array(row[6])
            
            # Cosine Similarity
            norm_db = np.linalg.norm(db_vec)
            if norm_target == 0 or norm_db == 0:
                score = 0
            else:
                score = np.dot(target_vec, db_vec) / (norm_target * norm_db)
            
            # Reconstruct result format: [headline, date, ticker, d1, d7, d30, score]
            scored_results.append([row[0], row[1], row[2], row[3], row[4], row[5], score])
            
        # Sort by similarity score (index 6) descending and take top k
        scored_results.sort(key=lambda x: x[6], reverse=True)
        return scored_results[:k]

    def _finnhub_fallback(self, headline, ticker, sentiment_score):
        """
        Fallback when no graph data available.
        Fetches related news from Finnhub API and analyzes on-the-fly.
        """
        import os
        import requests
        from datetime import datetime, timedelta
        
        api_key = os.getenv('FINNHUB_API_KEY')
        
        if not api_key:
            logger.warning("No Finnhub API key. Using sentiment-only prediction.")
            return self._sentiment_only_prediction(headline, sentiment_score)
        
        logger.info("📡 Fetching related news from Finnhub API...")
        
        # Try to extract ticker from headline if not provided
        tickers_to_try = []
        if ticker:
            tickers_to_try.append(ticker)
        else:
            # Check for common company mentions
            company_map = {
                'apple': 'AAPL', 'tesla': 'TSLA', 'google': 'GOOGL', 'amazon': 'AMZN',
                'microsoft': 'MSFT', 'meta': 'META', 'nvidia': 'NVDA', 'netflix': 'NFLX',
                'jpmorgan': 'JPM', 'visa': 'V', 'johnson': 'JNJ', 'reliance': 'RELIANCE.NS',
                'tata': 'TATAMOTORS.NS', 'infosys': 'INFY', 'hdfc': 'HDFCBANK.NS',
                'nifty': '^NSEI', 'sensex': '^BSESN', 'indian market': '^NSEI'
            }
            headline_lower = headline.lower()
            for name, sym in company_map.items():
                if name in headline_lower:
                    tickers_to_try.append(sym)
            
            # Default to market index if no company found
            if not tickers_to_try:
                tickers_to_try = ['AAPL', 'MSFT', 'GOOGL']  # Major tech as proxy
        
        all_news = []
        for sym in tickers_to_try[:2]:  # Limit to 2 tickers
            try:
                # Fetch news from last 7 days
                end_date = datetime.now()
                start_date = end_date - timedelta(days=7)
                
                url = f"https://finnhub.io/api/v1/company-news"
                params = {
                    'symbol': sym.replace('.NS', '').replace('^', ''),
                    'from': start_date.strftime('%Y-%m-%d'),
                    'to': end_date.strftime('%Y-%m-%d'),
                    'token': api_key
                }
                
                resp = requests.get(url, params=params, timeout=10)
                if resp.status_code == 200:
                    news = resp.json()[:5]  # Top 5 news per ticker
                    for item in news:
                        all_news.append({
                            'headline': item.get('headline', ''),
                            'source': item.get('source', 'Finnhub'),
                            'datetime': item.get('datetime', 0),
                            'ticker': sym
                        })
            except Exception as e:
                logger.warning(f"Finnhub fetch error for {sym}: {e}")
        
        if not all_news:
            logger.warning("No news from Finnhub. Using sentiment-only prediction.")
            return self._sentiment_only_prediction(headline, sentiment_score)
        
        # Analyze fetched news sentiment
        logger.info(f"Analyzing {len(all_news)} related news items from Finnhub...")
        
        sentiments = []
        evidence = []
        for item in all_news:
            blob = TextBlob(item['headline'])
            item_sentiment = blob.sentiment.polarity
            sentiments.append(item_sentiment)
            evidence.append({
                'headline': item['headline'],
                'date': datetime.fromtimestamp(item['datetime']).strftime('%Y-%m-%d') if item['datetime'] else 'Recent',
                'ticker': item['ticker'],
                'd1_return': None,
                'd7_return': None,
                'd30_return': None,
                'similarity_score': 0.7,  # Placeholder
                'source': 'Finnhub API (Live)'
            })
        
        # Calculate average sentiment
        avg_sentiment = np.mean(sentiments) if sentiments else sentiment_score
        combined_sentiment = (sentiment_score + avg_sentiment) / 2
        
        # Generate prediction based on sentiment
        if combined_sentiment > 0.2:
            sentiment_label = "POSITIVE"
            predicted_d1 = round(combined_sentiment * 2.5, 2)
        elif combined_sentiment < -0.2:
            sentiment_label = "NEGATIVE"
            predicted_d1 = round(combined_sentiment * 2.5, 2)
        else:
            sentiment_label = "NEUTRAL"
            predicted_d1 = round(combined_sentiment * 1.0, 2)
        
        prediction = {
            "sentiment": sentiment_label,
            "confidence": "MEDIUM (Finnhub Fallback)",
            "ai_sentiment": sentiment_score,
            "market_sentiment": avg_sentiment,
            "predicted_d1": predicted_d1,
            "predicted_d7": round(predicted_d1 * 1.5, 2),
            "predicted_d30": round(predicted_d1 * 2.0, 2),
            "evidence": evidence,
            "source": "Finnhub API (No historical data in graph)"
        }
        
        logger.info(f"✅ Finnhub Fallback Prediction: {sentiment_label} | Day 1: {predicted_d1}%")
        return prediction
    
    def _sentiment_only_prediction(self, headline, sentiment_score):
        """
        Last resort prediction based only on TextBlob sentiment.
        Used when both graph and Finnhub are unavailable.
        """
        if sentiment_score > 0.2:
            sentiment_label = "POSITIVE"
            predicted_d1 = round(sentiment_score * 2.0, 2)
        elif sentiment_score < -0.2:
            sentiment_label = "NEGATIVE"
            predicted_d1 = round(sentiment_score * 2.0, 2)
        else:
            sentiment_label = "NEUTRAL"
            predicted_d1 = 0.0
        
        return {
            "sentiment": sentiment_label,
            "confidence": "LOW (Sentiment Only)",
            "ai_sentiment": sentiment_score,
            "predicted_d1": predicted_d1,
            "predicted_d7": round(predicted_d1 * 1.2, 2),
            "predicted_d30": round(predicted_d1 * 1.5, 2),
            "evidence": [],
            "source": "TextBlob Sentiment Analysis (No external data)"
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict market impact of a news headline.")
    parser.add_argument("headline", type=str, help="The news headline to analyze")
    parser.add_argument("--ticker", type=str, help="Optional ticker context", default=None)
    
    args = parser.parse_args()
    
    predictor = MarketPredictor()
    predictor.predict(args.headline, args.ticker)
