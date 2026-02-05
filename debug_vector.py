import os
import sys
import numpy as np
from dotenv import load_dotenv

# Ensure import paths
sys.path.append(os.getcwd())
load_dotenv()

try:
    from graph.falkor_client import FalkorClient
except ImportError:
    from graph.falkor_client import FalkorClient

def debug_vector_search():
    client = FalkorClient()
    
    print("1. Recreating Index...")
    try:
        try:
            client.query("DROP INDEX NewsEvent")
            print("Dropped old index.")
        except:
            print("Index did not exist.")

        create_query = "CREATE VECTOR INDEX FOR (n:NewsEvent) ON (n.embedding) OPTIONS {dimension: 384, similarityFunction: 'cosine'}"
        client.query(create_query)
        print("Index Created New.")
            
    except Exception as e:
        print(f"Failed to create index: {e}")

    print("\n2. Testing Vector Query (Conversion Functions)...")
    try:
        # Create float32 vector as list
        vec_list = np.random.rand(384).astype(np.float32).tolist()
        params = {'vec': vec_list}
        
        # Sig: vecf32($vec)
        query = """CALL db.idx.vector.queryNodes('NewsEvent', 'embedding', 5, vecf32($vec)) YIELD node, score RETURN node.headline, score"""
        
        try:
            print("Testing with vecf32($vec)...")
            res = client.query(query, params)
            print("SUCCESS! Results:", res)
        except Exception as e:
            print(f"FAILED (vecf32): {e}")

    except Exception as e:
        print(f"Setup Error: {e}")

if __name__ == "__main__":
    debug_vector_search()
