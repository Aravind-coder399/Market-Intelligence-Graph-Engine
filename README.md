# Market Intelligence Graph Engine

A real-time market intelligence system that uses a knowledge graph to surface relevant historical events and help users make informed trading decisions.

## 🚀 Features

### Core Capabilities
- **Knowledge Graph** - FalkorDB-powered graph storing news, companies, and market relationships
- **Vector Similarity Search** - Find semantically similar past events using sentence embeddings
- **Real-Time News Ingestion** - Automated ingestion from Finnhub and FNSPID dataset
- **Portfolio Tracking** - Track holdings with live price updates

### Research Assistant
- Surface relevant historical events without making predictions
- Let users make their own informed decisions
- Compare multiple news headlines side-by-side

### Alert System
- Set up pattern-based alerts: "Notify me when news similar to X appears"
- Email/Slack notifications (configurable)
- Watchlist management

### Analytics Dashboard
- Real-time market trends for Top 10 companies
- 30-day price trajectory charts
- Sentiment distribution analysis

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Graph Database**: FalkorDB
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
- **Market Data**: yfinance
- **News API**: Finnhub
- **Cache**: Redis (optional)
- **LLM**: OpenAI GPT (optional)

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/Aravind-coder399/Market-Intelligence-Graph-Engine.git
cd Market-Intelligence-Graph-Engine

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## ⚙️ Configuration

Create a `.env` file with your API keys:

```env
FINNHUB_API_KEY=your_finnhub_key
OPENAI_API_KEY=your_openai_key  # Optional
REDIS_URL=redis://localhost:6379  # Optional

# FalkorDB Cloud
FALKORDB_HOST=your_host
FALKORDB_PORT=16752
FALKORDB_PASSWORD=your_password
```

## 🚀 Running the App

```bash
# Start Streamlit app
streamlit run main.py

# Run scheduled ingestion (for cron/task scheduler)
python scheduled_ingest.py
```

## 📊 Data Sources

- **FNSPID Dataset**: 10 years of financial news (Hugging Face)
- **Finnhub**: Real-time market news
- **yfinance**: Historical and real-time stock prices

## 🔧 Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Finnhub    │────▶│  Ingestion   │────▶│  FalkorDB    │
│   FNSPID     │     │  Pipeline    │     │  Knowledge   │
│   yfinance   │     │              │     │  Graph       │
└──────────────┘     └──────────────┘     └──────────────┘
                                                 │
                                                 ▼
                     ┌──────────────┐     ┌──────────────┐
                     │  Streamlit   │◀────│   Vector     │
                     │  Dashboard   │     │   Search     │
                     └──────────────┘     └──────────────┘
```

## 📁 Project Structure

```
stock_project/
├── main.py                 # Streamlit app entry point
├── predict_market.py       # Market prediction logic
├── llm_summary.py          # LLM-powered summaries
├── cache.py                # Redis caching layer
├── alerts.py               # Alert system
├── backtesting.py          # Strategy backtesting
├── scheduled_ingest.py     # Scheduled data ingestion
├── agents/
│   └── workflow.py         # Agent workflow
├── graph/
│   └── falkor_client.py    # FalkorDB client
├── ingestion/
│   ├── fnspid_loader.py    # FNSPID dataset loader
│   ├── graph_loader.py     # Graph data loader
│   └── news_loader.py      # Live news loader
└── requirements.txt
```

## 📝 License

MIT License

## 👤 Author

Aravind Appusamy
