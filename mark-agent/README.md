# mark-agent

Voice-driven trading strategy assistant.

## Loop

`listen → route_command (Claude) → execute → speak`  
Keyword routing is used automatically if the Claude API call fails.

## Setup

```bash
cd mark-agent
pip install -r requirements.txt
cp .env.example .env
# put your key in .env:
# ANTHROPIC_API_KEY=sk-ant-...
```

On Mac, PortAudio is required for the microphone:

```bash
brew install portaudio
pip install PyAudio
```

## Run

```bash
python main.py
```

## Voice commands (Hindi / English)

| Intent | Example phrases |
|--------|-----------------|
| Fetch data | डेटा लाओ, fetch, market |
| Generate strategy | रणनीति बनाओ, generate strategy |
| Backtest | बैकटेस्ट करो, backtest |
| Validate | सत्यापन, validate, overfit |
| Quit | बंद, quit, exit |

Claude chooses among: `fetch_data` / `generate_strategy` / `backtest` / `validate` / `general_answer`.  
Each Claude call prints input/output tokens and estimated cost.

## Modules

- `voice/listener.py` — mic → text (Hindi via SpeechRecognition)
- `voice/speaker.py` — text → speech (pyttsx3, Hindi voice when available)
- `data/binance_fetch.py` — Binance OHLCV → pandas DataFrame
- `strategy/generator.py` — proposes one rule-based strategy dict
- `strategy/backtester.py` — win_rate, max_drawdown, avg_R, trades, sharpe
- `strategy/validator.py` — 70/30 split; flags OVERFIT if train WR > test WR by 15%
- `brain/router.py` — Claude routing + keyword fallback
