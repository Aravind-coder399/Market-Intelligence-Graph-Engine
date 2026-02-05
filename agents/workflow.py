from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.researcher import ResearcherAgent
from agents.reporter import ReporterAgent

def create_graph():
    """
    Constructs the LangGraph workflow for Market Intelligence.
    """
    # 1. Initialize Agents
    researcher = ResearcherAgent()
    reporter = ReporterAgent()

    # 2. Initialize StateGraph
    workflow = StateGraph(AgentState)

    # 3. Add Nodes
    # Using lambda wrappers or method pointers since run methods match the signature State -> State (or close)
    # LangGraph expects func(state) -> new_state
    workflow.add_node("researcher", researcher.run)
    workflow.add_node("reporter", reporter.run)

    # 4. Define Edges
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "reporter")
    workflow.add_edge("reporter", END)

    # 5. Compile
    app = workflow.compile()
    
    return app

if __name__ == "__main__":
    # Test Run
    import logging
    print("Initializing Workflow...")
    app = create_graph()
    
    test_input = {
        "news_item": {
            "headline": "Tech stocks rally as inflation cools down",
            "date": "2023-01-01",
            "source": "Test"
        }
    }
    
    print("Running Workflow with Test Input...")
    try:
        result = app.invoke(test_input)
        print("\n--- Workflow Result ---")
        print(f"Analysis: {result.get('analysis')}")
        print(f"Similar Events Found: {len(result.get('similar_events', []))}")
    except Exception as e:
        print(f"Workflow Failed: {e}")
