import streamlit as st
import logging
import sys
import os
import pandas as pd
import yfinance as yf

# Ensure proper path for modules
sys.path.append(os.path.dirname(__file__))

try:
    from agents.workflow import create_graph
    from predict_market import MarketPredictor
    from llm_summary import LLMSummarizer, compare_headlines
    from cache import get_cache
    from alerts import get_alert_manager
    from backtesting import Backtester
    import graphviz
except ImportError as e:
    st.error(f"Failed to import modules: {e}")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Market Intelligence Graph", layout="wide")

# --- Session State Initialization ---
if 'selected_news' not in st.session_state:
    st.session_state.selected_news = None
if 'auto_analyze' not in st.session_state:
    st.session_state.auto_analyze = False
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {}
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'compare_mode' not in st.session_state:
    st.session_state.compare_mode = False
if 'compare_headlines' not in st.session_state:
    st.session_state.compare_headlines = []

# Target Companies
TARGET_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'JNJ']

TICKER_NAMES = {
    'AAPL': 'Apple', 'MSFT': 'Microsoft', 'GOOGL': 'Alphabet',
    'AMZN': 'Amazon', 'TSLA': 'Tesla', 'META': 'Meta',
    'NVDA': 'NVIDIA', 'JPM': 'JPMorgan', 'V': 'Visa', 'JNJ': 'Johnson & Johnson'
}

# --- Helper Functions ---
@st.cache_resource
def get_falkor_client():
    return FalkorClient()

# Import NewsLoader for Live Data
try:
    from ingestion.news_loader import NewsLoader
except ImportError:
    st.error("Failed to import NewsLoader")

@st.cache_resource
def get_news_loader():
    return NewsLoader()

def get_db_stats():
    """Get database statistics from FalkorDB."""
    try:
        from graph.falkor_client import FalkorClient
        client = FalkorClient()
        news_count = client.query("MATCH (n:News) RETURN count(n)")[0][0]
        company_count = client.query("MATCH (c:Company) RETURN count(c)")[0][0]
        topic_count = client.query("MATCH (t:Topic) RETURN count(t)")[0][0]
        return {"news": news_count, "companies": company_count, "topics": topic_count}
    except:
        return {"news": 0, "companies": 0, "topics": 0}

def calculate_portfolio_value():
    """Calculate current portfolio value."""
    total = 0
    details = []
    for ticker, shares in st.session_state.portfolio.items():
        try:
            stock = yf.Ticker(ticker)
            price = stock.history(period="1d")['Close'].iloc[-1]
            value = shares * price
            total += value
            details.append({"Ticker": ticker, "Shares": shares, "Price": f"${price:.2f}", "Value": f"${value:.2f}"})
        except:
            pass
    return total, details

@st.cache_data(ttl=1800, show_spinner=False)  # Run every 30 minutes
def auto_ingest_realtime_news():
    """Automatically ingest real-time news into Knowledge Graph on app load."""
    try:
        from ingestion.graph_loader import GraphLoader
        from ingestion.news_loader import NewsLoader
        
        loader = NewsLoader()
        graph_loader = GraphLoader()
        
        # Fetch live news from Finnhub
        live_news = loader.fetch_live_news()
        
        if live_news:
            # Format for graph ingestion
            records = []
            for item in live_news:
                if item.get('headline') and item.get('ticker'):
                    records.append({
                        'headline': item['headline'],
                        'date': item.get('date', 'today'),
                        'ticker': item['ticker'],
                        'source': item.get('source', 'Finnhub Live')
                    })
            
            if records:
                graph_loader.load_news(records)
                return len(records)
        return 0
    except Exception as e:
        logger.warning(f"Auto-ingest failed: {e}")
        return 0

def get_live_news():
    loader = get_news_loader()
    # Only Top 10 Companies
    target_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'JNJ']
    
    # Try fetching recently ingested items from graph as "Live Feed" simulation
    client = get_falkor_client()
    query = f"""
    MATCH (n:News) 
    WHERE n.ticker IN {str(target_tickers)}
    RETURN n.headline, n.date, n.ticker, n.source
    ORDER BY n.date DESC LIMIT 15
    """
    try:
        data = client.query(query)
        if data:
            return [{"headline": r[0], "date": r[1], "ticker": r[2], "source": r[3]} for r in data]
    except:
        pass
        
    return loader.fetch_live_news()

# --- Auto-Ingest on App Startup ---
with st.spinner("🔄 Syncing real-time news to Knowledge Base..."):
    ingested_count = auto_ingest_realtime_news()
    if ingested_count > 0:
        st.toast(f"✅ Ingested {ingested_count} real-time news articles", icon="📰")

# Get selected news from session state
selected_news_item = st.session_state.selected_news

# --- Sidebar with Multiple Sections ---
sidebar_tab = st.sidebar.radio("📋 Menu", ["📰 News Feed", "💼 Portfolio", "🗄️ Database", "🔔 Alerts", "📊 Backtest"], horizontal=True)

if sidebar_tab == "📰 News Feed":
    st.sidebar.caption("Real-time Top 10 Company News")
    
    # Fetch live news
    recent_news = get_live_news()
    
    # Group by Ticker
    grouped_news = {}
    if recent_news:
        for item in recent_news:
            t = item.get('ticker')
            if t:
                if t not in grouped_news: grouped_news[t] = []
                grouped_news[t].append(item)
    
    # Selection Mode
    category = st.sidebar.selectbox("Filter by Company", ["All"] + list(grouped_news.keys()))
    
    # Build Options
    if category == "All":
        display_items = recent_news if recent_news else []
    else:
        display_items = grouped_news.get(category, [])
    
    if display_items:
        st.sidebar.markdown(f"**{len(display_items)} articles**")
        for idx, item in enumerate(display_items):
            btn_label = f"{item['ticker']} | {item['headline'][:30]}..."
            if st.sidebar.button(btn_label, key=f"news_btn_{idx}", use_container_width=True):
                st.session_state.selected_news = item
                st.session_state.auto_analyze = True
                st.rerun()
    else:
        st.sidebar.info("No news available.")

elif sidebar_tab == "💼 Portfolio":
    st.sidebar.caption("Track your holdings")
    
    # Add to portfolio
    with st.sidebar.form("add_holding"):
        col1, col2 = st.columns(2)
        ticker = col1.selectbox("Ticker", TARGET_TICKERS, key="portfolio_ticker")
        shares = col2.number_input("Shares", min_value=1, value=10, key="portfolio_shares")
        if st.form_submit_button("➕ Add", use_container_width=True):
            if ticker in st.session_state.portfolio:
                st.session_state.portfolio[ticker] += shares
            else:
                st.session_state.portfolio[ticker] = shares
            st.rerun()
    
    # Display portfolio
    if st.session_state.portfolio:
        total, details = calculate_portfolio_value()
        st.sidebar.metric("💰 Total Value", f"${total:,.2f}")
        st.sidebar.dataframe(pd.DataFrame(details), hide_index=True, use_container_width=True)
        
        if st.sidebar.button("🗑️ Clear Portfolio", use_container_width=True):
            st.session_state.portfolio = {}
            st.rerun()
    else:
        st.sidebar.info("No holdings yet. Add stocks above!")

elif sidebar_tab == "🗄️ Database":
    st.sidebar.caption("Knowledge Graph Statistics")
    stats = get_db_stats()
    
    st.sidebar.metric("📰 News Articles", f"{stats['news']:,}")
    st.sidebar.metric("🏢 Companies", f"{stats['companies']:,}")
    st.sidebar.metric("🏷️ Topics", f"{stats['topics']:,}")
    
    st.sidebar.divider()
    st.sidebar.caption("Analysis History")
    if st.session_state.analysis_history:
        for h in st.session_state.analysis_history[-5:]:
            st.sidebar.text(f"• {h[:40]}...")
    else:
        st.sidebar.info("No analysis yet.")

elif sidebar_tab == "🔔 Alerts":
    st.sidebar.caption("Pattern-based monitoring")
    
    alert_mgr = get_alert_manager()
    
    # Add new alert
    with st.sidebar.form("add_alert"):
        pattern = st.text_input("Watch for news like:", placeholder="Apple earnings beat...")
        alert_ticker = st.selectbox("Ticker (optional)", ["Any"] + TARGET_TICKERS)
        threshold = st.slider("Similarity Threshold", 0.5, 0.95, 0.7)
        if st.form_submit_button("➕ Add Alert", use_container_width=True):
            alert_mgr.add_alert(
                pattern, 
                ticker=None if alert_ticker == "Any" else alert_ticker,
                threshold=threshold
            )
            st.rerun()
    
    # Show existing alerts
    alerts = alert_mgr.get_alerts()
    if alerts:
        st.sidebar.markdown(f"**{len(alerts)} active alerts**")
        for alert in alerts:
            with st.sidebar.expander(f"🔔 {alert['pattern'][:25]}..."):
                st.text(f"Ticker: {alert.get('ticker', 'Any')}")
                st.text(f"Threshold: {alert['threshold']}")
                st.text(f"Triggered: {alert['triggered_count']}x")
                if st.button("🗑️ Remove", key=f"rm_alert_{alert['id']}"):
                    alert_mgr.remove_alert(alert['id'])
                    st.rerun()
    else:
        st.sidebar.info("No alerts set. Add one above!")

elif sidebar_tab == "📊 Backtest":
    st.sidebar.caption("Test prediction accuracy")
    
    if st.sidebar.button("🧪 Run Backtest", use_container_width=True):
        with st.spinner("Running backtest..."):
            predictor = MarketPredictor()
            backtester = Backtester(predictor)
            results = backtester.run_backtest_from_graph(limit=50)
            
            if 'error' not in results:
                st.sidebar.metric("Accuracy", f"{results['accuracy']}%")
                st.sidebar.metric("MAE", f"{results['mean_absolute_error']}%")
                st.sidebar.success(f"Tested {results['total_predictions']} predictions")
            else:
                st.sidebar.error(results['error'])


# --- Main Content ---
st.title("🚀 Market Intelligence Graph Engine")
st.markdown("""
**GraphRAG System** correlating news with market history.
""")

# Input Data
default_headline = ""
default_date = "today"
default_ticker = ""

# If user selected from sidebar, populate variables
if selected_news_item:
    default_headline = selected_news_item.get('headline', '')
    default_date = pd.to_datetime(selected_news_item.get('date', 'today'))
    default_ticker = selected_news_item.get('ticker', '')
    
    # Display "Ground Truth" ONLY if historical data exists (i.e., not a live item)
    # Live items from NewsLoader won't have 'day_1' key or it will be N/A.
    # Historical items from Graph query have 'day_1' key.
    if 'day_1' in selected_news_item:
        with st.expander("🔍 Ground Truth (Actual History)", expanded=True):
            st.info(f"**Selected Event**: {default_headline} ({default_date.date()})")
            c1, c2, c3 = st.columns(3)
            c1.metric("Actual Day 1", f"{selected_news_item.get('day_1', 0)}%")
            c2.metric("Actual Day 7", f"{selected_news_item.get('day_7', 0)}%")
            c3.metric("Actual Day 30", f"{selected_news_item.get('day_30', 0)}%")
    else:
        # For live news, we can't show ground truth returns
        st.caption("ℹ️ Selected Live News (No historical returns available yet)")

# Input Form
with st.form("news_form"):
    # Mode Toggle
    compare_enabled = st.checkbox("🔀 Compare Mode (Analyze 2 headlines)", value=st.session_state.compare_mode)
    
    headline = st.text_area("News Headline 1", value=default_headline, height=80)
    
    # Second headline for compare mode
    headline2 = ""
    ticker2 = ""
    if compare_enabled:
        headline2 = st.text_area("News Headline 2", height=80)
    
    col_d, col_t = st.columns(2)
    date = col_d.date_input("Date", value=default_date)
    ticker = col_t.text_input("Ticker (Optional)", value=default_ticker)
    
    if compare_enabled:
        ticker2 = st.text_input("Ticker 2 (Optional)")
    
    col_submit, col_llm = st.columns([2, 1])
    submitted = col_submit.form_submit_button("🔍 Analyze Market Impact", use_container_width=True)
    generate_llm = col_llm.form_submit_button("🤖 AI Report", use_container_width=True)

# Update compare mode state
st.session_state.compare_mode = compare_enabled

# Handle auto-analyze from sidebar click
if st.session_state.auto_analyze:
    submitted = True
    st.session_state.auto_analyze = False

# --- Real-Time Market Trends Section ---
st.divider()
st.subheader("📊 Real-Time Market Trends (Top 10 Companies)")

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_market_trends():
    """Fetch real-time market data for target tickers."""
    try:
        data = []
        for ticker in TARGET_TICKERS:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="5d")
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                change_pct = ((current_price - prev_price) / prev_price) * 100
                
                data.append({
                    "Ticker": ticker,
                    "Company": info.get('shortName', ticker),
                    "Price": f"${current_price:.2f}",
                    "Change %": round(change_pct, 2),
                    "Volume": f"{hist['Volume'].iloc[-1]:,.0f}"
                })
        return pd.DataFrame(data)
    except Exception as e:
        return pd.DataFrame()

with st.spinner("Loading market data..."):
    market_df = get_market_trends()
    if not market_df.empty:
        # Display as metrics in columns
        cols = st.columns(5)
        for idx, row in market_df.iterrows():
            col_idx = idx % 5
            cols[col_idx].metric(
                label=row['Ticker'],
                value=row['Price'],
                delta=f"{row['Change %']}%"
            )
        
        # Show detailed table in expander
        with st.expander("View Detailed Market Data"):
            st.dataframe(market_df, use_container_width=True)
    else:
        st.info("Market data temporarily unavailable.")

# --- Moving Stock Price Charts ---
st.subheader("📈 Stock Price Trends (30 Days)")

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_price_history(tickers, period="1mo"):
    """Fetch historical price data for multiple tickers."""
    try:
        data = yf.download(tickers, period=period, progress=False)['Close']
        return data
    except:
        return pd.DataFrame()

selected_tickers = st.multiselect(
    "Select companies to compare:",
    TARGET_TICKERS,
    default=['AAPL', 'MSFT', 'NVDA']
)

if selected_tickers:
    with st.spinner("Loading price history..."):
        price_df = get_price_history(selected_tickers)
        if not price_df.empty:
            # Normalize to 100 for comparison
            normalized = (price_df / price_df.iloc[0]) * 100
            st.line_chart(normalized)
            st.caption("Normalized to 100 at start for comparison")
        else:
            st.warning("Could not fetch price history.")
else:
    st.info("Select at least one company to view price trends.")

st.divider()

# Handle both submit buttons
should_analyze = (submitted or generate_llm) and headline

if should_analyze:
    with st.spinner("🤖 Analyzing Knowledge Graph..."):
        try:
            predictor = MarketPredictor()
            llm = LLMSummarizer()
            
            # Compare Mode
            if compare_enabled and headline2:
                st.subheader("🔀 Comparison Analysis")
                
                comparison = compare_headlines(
                    predictor, 
                    [headline, headline2], 
                    [ticker, ticker2]
                )
                
                # Display comparison table
                comp_df = pd.DataFrame(comparison["comparisons"])
                st.dataframe(comp_df, use_container_width=True)
                
                # Summary
                st.markdown(comparison["summary"])
                
                # LLM Summary for each if requested
                if generate_llm:
                    st.subheader("🤖 AI Analysis Reports")
                    for i, res in enumerate(comparison["comparisons"]):
                        with st.expander(f"Report {i+1}: {res['headline'][:50]}..."):
                            result_dict = {
                                "sentiment": res["sentiment"],
                                "confidence": res["confidence"],
                                "ai_sentiment": res["ai_sentiment"],
                                "predicted_d1": res["predicted_d1"],
                                "predicted_d30": res["predicted_d30"],
                                "evidence": []
                            }
                            summary = llm.generate_summary(result_dict, res["headline"], res["ticker"])
                            st.info(summary)
            else:
                # Single headline analysis
                result = predictor.predict(headline, ticker=ticker)
                
                if not result:
                    st.warning("No relevant analysis could be generated.")
                else:
                    sentiment = result.get("sentiment", "NEUTRAL")
                    confidence = result.get("confidence", "LOW")
                    ai_score = result.get("ai_sentiment", 0.0)
                
                d1_est = result.get("predicted_d1", 0.0)
                d7_est = result.get("predicted_d7", 0.0)
                d30_est = result.get("predicted_d30", 0.0)
                summary = result.get("summary", "")
                
                # Header
                st.subheader(f"🔮 Market Prediction: {sentiment}")
                if "Insufficient" in summary:
                    st.warning(summary)
                else:
                    st.success(f"{summary}")

                # Metrics Row
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Confidence", confidence, help="High if History and AI agree")
                col2.metric("AI Text Sentiment", f"{ai_score:.2f}", help="-1.0 (Neg) to 1.0 (Pos)")
                col3.metric("Est. 1-Day Return", f"{d1_est}%", delta=d1_est)
                col4.metric("Est. 30-Day Return", f"{d30_est}%", delta=d30_est)
                
                # LLM Generated Summary
                if generate_llm:
                    with st.expander("🤖 AI-Generated Analysis Report", expanded=True):
                        with st.spinner("Generating AI report..."):
                            ai_report = llm.generate_summary(result, headline, ticker)
                            st.info(ai_report)
                
                # Save to analysis history
                st.session_state.analysis_history.append(headline)
                
                # Export Button
                evidence = result.get("evidence", [])
                if evidence:
                    export_data = pd.DataFrame(evidence)
                    csv = export_data.to_csv(index=False)
                    st.download_button(
                        "📥 Export Analysis (CSV)",
                        csv,
                        file_name=f"analysis_{ticker or 'custom'}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                
                st.divider()
                
                # --- Visualizations ---
                tab_traj, tab_analytics, tab_net, tab_proof = st.tabs(["📈 Return Trajectory", "📊 Analytics Dashboard", "🕸️ Knowledge Graph", "📚 Evidence Table"])
                
                with tab_traj:
                    st.caption("Projected average market reaction over 30 days based on similar events.")
                    chart_data = pd.DataFrame({
                        "Days": [0, 1, 7, 30],
                        "Return %": [0.0, d1_est, d7_est, d30_est]
                    }).set_index("Days")
                    st.line_chart(chart_data)
                    
                    # Add confidence interval visualization
                    evidence = result.get("evidence", [])
                    if evidence:
                        st.markdown("**Return Distribution Across Evidence**")
                        returns_data = [ev.get('d1_return') for ev in evidence if ev.get('d1_return') is not None]
                        if returns_data:
                            st.bar_chart(pd.DataFrame({"Historical Day 1 Returns": returns_data}))
                
                with tab_analytics:
                    st.caption("Deep dive into prediction trustworthiness and evidence quality.")
                    evidence = result.get("evidence", [])
                    
                    if evidence:
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.markdown("**Similarity Score Distribution**")
                            sim_scores = [ev['similarity_score'] for ev in evidence]
                            st.bar_chart(pd.DataFrame({"Similarity": sim_scores}))
                            avg_sim = sum(sim_scores) / len(sim_scores)
                            st.metric("Avg Similarity", f"{avg_sim:.3f}")
                        
                        with col_b:
                            st.markdown("**Return vs Similarity Correlation**")
                            scatter_data = []
                            for ev in evidence:
                                if ev.get('d1_return') is not None:
                                    scatter_data.append({
                                        "Similarity": ev['similarity_score'],
                                        "Day 1 Return %": ev['d1_return']
                                    })
                            if scatter_data:
                                scatter_df = pd.DataFrame(scatter_data)
                                st.scatter_chart(scatter_df.set_index("Similarity"))
                        
                        # Confidence Breakdown
                        st.divider()
                        st.markdown("**Confidence Analysis**")
                        conf_col1, conf_col2, conf_col3 = st.columns(3)
                        
                        with conf_col1:
                            st.metric("Evidence Count", len(evidence), help="More evidence = Higher confidence")
                        with conf_col2:
                            hist_align = "✅ Aligned" if (d1_est > 0 and ai_score > 0) or (d1_est < 0 and ai_score < 0) else "⚠️ Divergent"
                            st.metric("History ↔ AI", hist_align)
                        with conf_col3:
                            consistency = "High" if all(ev.get('d1_return', 0) * d1_est > 0 for ev in evidence if ev.get('d1_return') is not None) else "Mixed"
                            st.metric("Direction Consistency", consistency)
                    else:
                        st.info("No evidence available for analytics.")
                    
                with tab_net:
                    st.caption("Visualizing the connection between the Context (Company) and the Evidence (News).")
                    try:
                        g = graphviz.Digraph()
                        g.attr(rankdir='LR')
                        
                        # Central Node
                        center_label = ticker if ticker else "Search"
                        g.node('ROOT', center_label, shape='doublecircle', color='blue', style='filled', fillcolor='lightblue')
                        
                        evidence = result.get("evidence", [])
                        for i, ev in enumerate(evidence):
                            # News Nodes
                            node_id = f"NEWS_{i}"
                            # Shorten headline for display
                            short_hl = (ev['headline'][:30] + '..') if len(ev['headline']) > 30 else ev['headline']
                            score = ev['similarity_score']
                            
                            # Color based on return
                            d1 = ev.get('d1_return')
                            color = 'green' if d1 and d1 > 0 else ('red' if d1 and d1 < 0 else 'grey')
                            
                            g.node(node_id, f"{short_hl}\n(Sim: {score:.2f})", shape='box', color=color)
                            g.edge('ROOT', node_id, label="HAS_EVIDENCE")
                            
                        st.graphviz_chart(g)
                    except Exception as g_err:
                        st.warning(f"Graph visualization unavailable (Graphviz binary missing?): {g_err}")

                with tab_proof:
                    st.caption("Detailed list of historical events used for this prediction.")
                    evidence = result.get("evidence", [])
                    if evidence:
                        table_data = []
                        for ev in evidence:
                            table_data.append({
                                "Date": ev['date'],
                                "Ticker": ev['ticker'],
                                "Headline": ev['headline'],
                                "Similarity": f"{ev['similarity_score']:.2f}",
                                "Day 1 %": f"{ev['d1_return']:.2f}%" if ev['d1_return'] is not None else "-",
                                "Day 30 %": f"{ev['d30_return']:.2f}%" if ev.get('d30_return') is not None else "-"
                            })
                        st.dataframe(table_data, use_container_width=True)
                    else:
                        st.info("No direct historical matches found.")

        except Exception as e:
            st.error(f"System Error: {e}")
            logger.error(f"System Error: {e}", exc_info=True)

elif submitted and not headline:
    st.warning("Please enter a headline.")

# Sidebar Footer
with st.sidebar:
    st.divider()
    st.caption("System Status: Online 🟢")
