# YouTube Stock Analysis with Transcript Processing

This project implements an automated YouTube stock analysis system that analyzes financial content from YouTube channels using CrewAI, DeepSeek's local LLM, and BrightData.
- [Bright Data](https://brdta.com/dailydoseofds) is used to scrape YouTube channels for video metadata and transcripts
- [CrewAI](https://github.com/joaomdmoura/crewAI) powers an intelligent multi-agent system for stock analysis and insights
- [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) model runs locally for private and efficient AI processing
- [Streamlit](https://streamlit.io/) provides an intuitive web interface for viewing stock mentions and analysis

## How it Works

This project runs entirely on your local machine, leveraging Ollama for AI processing:

1. **Data Collection**: 
   - Uses BrightData's dataset API to scrape YouTube channels
   - Collects video metadata (titles, descriptions, dates, view counts)
   - Retrieves video transcripts when available
   - Supports both channel-based and keyword-based scraping
   - Handles date ranges and post limits for targeted collection

2. **Local AI Processing**: 
   - Uses Ollama to run the [DeepSeek-R1](https://www.deepseek.com/) model locally on your laptop
   - Based on the [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) architecture, optimized for reasoning
   - No cloud dependencies or API costs for AI processing
   - Complete privacy of your analysis data
   - Powered by DeepSeek's state-of-the-art 7B parameter language model

3. **Multi-Agent System**:
   - Built with [CrewAI](https://www.crewai.io/), a cutting-edge framework for orchestrating role-playing AI agents
   - Agents work collaboratively to analyze financial content
   - Leverages CrewAI's task management and agent communication system
   - Built on the principles of [AutoGPT](https://docs.crewai.com/how-it-works/) for autonomous operation

4. **Agents and Their Roles**:
   - **Analysis Agent**: A stock market expert that analyzes YouTube content to:
     - Extract information from video titles and descriptions
     - Process available video transcripts for deeper insights
     - Identify mentions of publicly traded companies
     - Match companies with their correct ticker symbols
     - Analyze sentiment (bullish/bearish) around each mention
     - Capture specific price targets and predictions
     - Identify sector trends and related stocks

   - **Response Synthesizer Agent**: A financial data organizer that:
     - Creates structured lists of identified stocks with their ticker symbols
     - Groups stocks by sectors or themes
     - Summarizes sentiment and predictions
     - Highlights frequently mentioned or strongly emphasized stocks
     - Presents information in an easily digestible format

5. **Interactive Web Interface**:
   - Built with [Streamlit](https://streamlit.io/), the fastest way to build data apps
   - Real-time stock analysis visualization
   - Interactive data filtering and exploration
   - Clean, modern UI with [Streamlit Components](https://docs.streamlit.io/library/components)
   - Mobile-responsive design

## Use Cases

This analysis system can be adapted for various financial content sources:

1. **Financial YouTube Channels**:
   - Stock market commentary channels
   - Investment advice videos
   - Market analysis shows
   - Trading strategy tutorials

2. **Financial Podcasts**:
   - Investment podcasts with transcripts
   - Market news shows
   - Expert interview series
   - Trading psychology discussions

3. **Educational Content**:
   - Financial education channels
   - Investment courses
   - Trading workshops
   - Market analysis tutorials

4. **News and Updates**:
   - Financial news channels
   - Market update videos
   - Economic analysis content
   - Company earnings discussions

The system's architecture allows for easy adaptation to different content sources while maintaining the same powerful analysis capabilities.

## System Requirements

- **CPU**: Modern multi-core processor (recommended 4+ cores)
- **RAM**: Minimum 16GB RAM recommended
- **Storage**: At least 10GB free space for model and data
- **Operating System**: 
  - MacOS 12 or later
  - Linux (modern distribution)

---
## Setup and installations

**Get BrightData API Key**:
- Go to [Bright Data](https://brdta.com/dailydoseofds) and sign up for an account.
- Once you have an account, go to the API Key page and copy your API key.
- Paste your API key by creating a `.env` file as follows:

```
BRIGHT_DATA_API_KEY=your_api_key
```

**BrightData API Limits**:
- Maximum 50 videos per channel per request
- Recommended to use date ranges for channels with many videos
- Example configuration:
  ```python
  num_of_posts = 50  # Maximum videos per request
  start_date = "2024-01-01"  # Format: YYYY-MM-DD
  end_date = "2024-03-20"    # Format: YYYY-MM-DD
  ```
- For more videos, make multiple requests with different date ranges
- Rate limits apply based on your BrightData subscription tier

**Setup Ollama**:
   ```bash
   # For Linux:
   curl -fsSL https://ollama.com/install.sh | sh
   
   # For Mac:
   brew install ollama
   
   # Start Ollama (Mac only)
   ollama serve
   
   # Pull DeepSeek-R1 model (approximately 4GB download)
   # More info: https://github.com/deepseek-ai/DeepSeek-R1
   ollama pull deepseek-r1:7b 
   ```

**Install Dependencies**:
   Ensure you have Python 3.11 or later installed.
   ```bash
   # Install core dependencies
   pip install streamlit ollama crewai crewai-tools

   # For more information about dependencies:
   # Streamlit: https://docs.streamlit.io/
   # CrewAI: https://docs.crewai.com/
   # GitHub: https://github.com/joaomdmoura/crewAI
   ```

---

## Run the project

Launch the application:
   ```bash
   streamlit run app.py
   ```

Visit [http://localhost:8501](http://localhost:8501) to access the interactive web interface.

All analysis is performed locally on your machine, ensuring privacy and control over your data.

---

## License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.
