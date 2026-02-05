import sys
import os
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from graph.falkor_client import FalkorClient

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def debug_similarity():
    client = FalkorClient()
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Test Caaes: Two very different concepts
    q1 = "Apple iPhone sales reach record highs"
    q2 = "Oil prices plummet due to surplus supply"
    
    print(f"--- Query 1: {q1} ---")
    vec1 = encoder.encode(q1).tolist()
    res1 = run_query(client, vec1)
    
    print(f"\n--- Query 2: {q2} ---")
    vec2 = encoder.encode(q2).tolist()
    res2 = run_query(client, vec2)
    
    # Compare
    print("\n--- COMPARISON ---")
    top1 = res1[0][0] if res1 else "None"
    top2 = res2[0][0] if res2 else "None"
    
    print(f"Top Result 1: {top1}")
    print(f"Top Result 2: {top2}")
    
    if top1 == top2:
        print("❌ FAIL: Results are IDENTICAL. Vector search is not differentiating.")
    else:
        print("✅ PASS: Results are different.")

def run_query(client, embedding):
    query = """
    CALL db.idx.vector.queryNodes('News', 'embedding', 3, vecf32($vec)) 
    YIELD node, score
    RETURN node.headline, score
    """
    try:
        results = client.query(query, {'vec': embedding})
        for r in results:
            print(f"  [{r[1]:.4f}] {r[0][:50]}...")
        return results
    except Exception as e:
        print(f"Server Search failed: {e}")
        # Try fallback manual
        return manual_search(client, embedding)

def manual_search(client, target_vec):
    print("  (Using Manual Fallback)")
    query = "MATCH (n:News) WHERE n.embedding IS NOT NULL RETURN n.headline, n.embedding"
    rows = client.query(query)
    results = []
    t_vec = np.array(target_vec)
    norm_t = np.linalg.norm(t_vec)
    
    for r in rows:
        d_vec = np.array(r[1])
        norm_d = np.linalg.norm(d_vec)
        score = np.dot(t_vec, d_vec) / (norm_t * norm_d)
        results.append((r[0], score))
    
    results.sort(key=lambda x: x[1], reverse=True)
    top_k = results[:3]
    for r in top_k:
        print(f"  [{r[1]:.4f}] {r[0][:50]}...")
    return top_k

if __name__ == "__main__":
    debug_similarity()
