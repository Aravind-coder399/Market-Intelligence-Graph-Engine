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
            logger.warning("No similar historical events found.")
            return

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict market impact of a news headline.")
    parser.add_argument("headline", type=str, help="The news headline to analyze")
    parser.add_argument("--ticker", type=str, help="Optional ticker context", default=None)
    
    args = parser.parse_args()
    
    predictor = MarketPredictor()
    predictor.predict(args.headline, args.ticker)
