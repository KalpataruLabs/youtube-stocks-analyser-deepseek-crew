# YouTube Stock Analysis

Analyze financial content from YouTube channels using CrewAI, local LLMs, and BrightData:
- [BrightData](https://brdta.com/dailydoseofds): YouTube channel scraping
- [CrewAI](https://github.com/joaomdmoura/crewAI): Multi-agent system for analysis
- [Ollama](https://ollama.com): Local LLM processing
- [Streamlit](https://streamlit.io/): Web interface

## How it Works

1. **Data Collection**: BrightData API scrapes YouTube channels for video data and transcripts
   
2. **Local AI Processing**: 
   - Run models locally with Ollama (DeepSeek-R1, Llama2, Mistral, Phi3, Gemma3)
   - No cloud dependencies or API costs
   - Model selection via UI

3. **Multi-Agent System**:
   - **Analysis Agent**: Processes transcripts, identifies stocks, analyzes sentiment
   - **Response Synthesizer**: Organizes findings into structured reports

## System Requirements

- **CPU**: 4+ cores recommended
- **RAM**: 16GB+ recommended
- **Storage**: 10GB+ free space
- **OS**: MacOS 12+ or modern Linux

## Setup

1. **BrightData API Key**:
   - Sign up at [Bright Data](https://brdta.com/dailydoseofds)
   - Create `.env` file with: `BRIGHT_DATA_API_KEY=your_api_key`

2. **Ollama Setup**:
   ```bash
   # Linux
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Mac
   brew install ollama
   ollama serve
   
   # Download models
   ollama pull deepseek-r1:7b  # Default model
   
   # Optional additional models
   ollama pull llama2:7b
   ollama pull mistral:7b
   ollama pull phi3:3b
   ollama pull gemma3:7b
   ```

3. **Dependencies**:
   ```bash
   pip install streamlit ollama crewai crewai-tools
   ```

## Usage

```bash
streamlit run app.py
```

Visit [http://localhost:8501](http://localhost:8501) for the interface:
- Add YouTube channels
- Set date ranges
- Select LLM model
- Toggle transcript reuse

All analysis runs locally for complete privacy.

## License

[MIT License](LICENSE)
