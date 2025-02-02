from crewai_tools import FileReadTool
from crewai import Agent, Crew, Process, Task, LLM
import streamlit as st
import os
import tempfile
import gc
import base64
import time
import yaml

from tqdm import tqdm
from brightdata_scrapper import *
from dotenv import load_dotenv
load_dotenv()


docs_tool = FileReadTool()

bright_data_api_key = os.getenv("BRIGHT_DATA_API_KEY")


@st.cache_resource
def load_llm():
    llm = LLM(
        model="ollama/deepseek-r1:7b",
        base_url="http://localhost:11434"
    )
    return llm

# ===========================
#   Define Agents & Tasks
# ===========================


def create_agents_and_tasks():
    """Creates a Crew for analysis of the channel scrapped output"""

    with open("config.yaml", 'r') as file:
        config = yaml.safe_load(file)

    analysis_agent = Agent(
        role=config["agents"][0]["role"],
        goal=config["agents"][0]["goal"],
        backstory=config["agents"][0]["backstory"],
        verbose=True,
        tools=[docs_tool],
        llm=load_llm()
    )

    response_synthesizer_agent = Agent(
        role=config["agents"][1]["role"],
        goal=config["agents"][1]["goal"],
        backstory=config["agents"][1]["backstory"],
        verbose=True,
        llm=load_llm()
    )

    analysis_task = Task(
        description=config["tasks"][0]["description"],
        expected_output=config["tasks"][0]["expected_output"],
        agent=analysis_agent
    )

    response_task = Task(
        description=config["tasks"][1]["description"],
        expected_output=config["tasks"][1]["expected_output"],
        agent=response_synthesizer_agent
    )

    crew = Crew(
        agents=[analysis_agent, response_synthesizer_agent],
        tasks=[analysis_task, response_task],
        process=Process.sequential,
        verbose=True
    )
    return crew

# ===========================
#   Streamlit Setup
# ===========================

# Load images using context managers
def load_image_as_base64(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Load both images
crewai_image = load_image_as_base64("assets/crewai.png")
brightdata_image = load_image_as_base64("assets/brightdata.png")

st.markdown("""
    # YouTube Trend Analysis powered by <img src="data:image/png;base64,{}" width="120" style="vertical-align: -3px;"> & <img src="data:image/png;base64,{}" width="120" style="vertical-align: -3px;">
""".format(crewai_image, brightdata_image), unsafe_allow_html=True)


if "messages" not in st.session_state:
    st.session_state.messages = []  # Chat history

if "response" not in st.session_state:
    st.session_state.response = None

if "crew" not in st.session_state:
    st.session_state.crew = None      # Store the Crew object

# Initialize skip_transcripts in session state if not present
if "skip_transcripts" not in st.session_state:
    st.session_state.skip_transcripts = True

def reset_chat():
    st.session_state.messages = []
    gc.collect()

def get_video_id_from_data(video_data):
    """Extract video ID from video data using various possible fields"""
    video_id = None
    # Try discovery_input first
    if 'discovery_input' in video_data and isinstance(video_data['discovery_input'], dict):
        discovery_url = video_data['discovery_input'].get('url', '')
        if 'youtube.com/watch?v=' in discovery_url:
            video_id = discovery_url.split('v=')[1].split('&')[0]
    
    # If not found, try the direct url field
    if not video_id and 'url' in video_data:
        video_url = video_data['url']
        if isinstance(video_url, str):
            if 'youtube.com/watch?v=' in video_url:
                video_id = video_url.split('v=')[1].split('&')[0]
            elif 'youtu.be/' in video_url:
                video_id = video_url.split('youtu.be/')[1].split('?')[0]
    
    # If still not found, try input field
    if not video_id and 'input' in video_data and isinstance(video_data['input'], dict):
        input_url = video_data['input'].get('url', '')
        if 'youtube.com/watch?v=' in input_url:
            video_id = input_url.split('v=')[1].split('&')[0]
    
    return video_id

def start_analysis():
    # Create a status container for consistent updates
    status_container = st.empty()

    # Always scrape to get the list of videos for the date range
    with st.spinner('Getting video list for the selected date range...'):
        try:
            channel_snapshot_id = trigger_scraping_channels(
                bright_data_api_key, 
                st.session_state.youtube_channels, 
                10, 
                st.session_state.start_date, 
                st.session_state.end_date, 
                "Latest", 
                ""
            )
            
            if not channel_snapshot_id or (isinstance(channel_snapshot_id, dict) and channel_snapshot_id.get('status') == 'failed'):
                error_msg = channel_snapshot_id.get('error', 'Unknown error') if isinstance(channel_snapshot_id, dict) else "Failed to get video list"
                if 'limit' in error_msg.lower():
                    status_container.error(
                        "Video limit exceeded. BrightData allows maximum 50 videos per request. "
                        "Please use a smaller date range or fewer channels."
                    )
                else:
                    status_container.error(f"Failed to get video list: {error_msg}")
                return
                
            status = get_progress(bright_data_api_key, channel_snapshot_id['snapshot_id'])
            
            if not status:
                status_container.error("Failed to get status. Please check your internet connection.")
                return
            
            # Poll for status
            max_retries = 30
            retry_count = 0
            
            while status and status.get('status') != "ready" and retry_count < max_retries:
                status_container.info(f"Current status: {status.get('status', 'unknown')}")
                time.sleep(10)
                status = get_progress(bright_data_api_key, channel_snapshot_id['snapshot_id'])
                retry_count += 1
                
                if not status:
                    status_container.error("Lost connection to Brightdata API. Please try again.")
                    return
                    
                if status.get('status') == "failed":
                    status_container.error(f"Scraping failed: {status}")
                    return
            
            if retry_count >= max_retries or not status or status.get('status') != "ready":
                status_container.error("Failed to get video list. Please try again.")
                return

            # Get the video list
            channel_scrapped_output = get_output(bright_data_api_key, status['snapshot_id'], format="json")
            if not channel_scrapped_output or not isinstance(channel_scrapped_output, list) or len(channel_scrapped_output) == 0:
                st.error("No videos found for the selected date range")
                return

            videos_data = channel_scrapped_output[0] if isinstance(channel_scrapped_output[0], list) else channel_scrapped_output
            
            # Extract video IDs from the current results
            current_video_ids = []
            for video_data in videos_data:
                video_id = None
                # Try discovery_input first
                if 'discovery_input' in video_data and isinstance(video_data['discovery_input'], dict):
                    discovery_url = video_data['discovery_input'].get('url', '')
                    if 'youtube.com/watch?v=' in discovery_url:
                        video_id = discovery_url.split('v=')[1].split('&')[0]
                
                # If not found, try the direct url field
                if not video_id and 'url' in video_data:
                    video_url = video_data['url']
                    if isinstance(video_url, str):
                        if 'youtube.com/watch?v=' in video_url:
                            video_id = video_url.split('v=')[1].split('&')[0]
                        elif 'youtu.be/' in video_url:
                            video_id = video_url.split('youtu.be/')[1].split('?')[0]
                
                if video_id:
                    current_video_ids.append(video_id)

            # Check if we have all needed transcripts
            if st.session_state.skip_transcripts:
                if not os.path.exists("transcripts"):
                    os.makedirs("transcripts")
                
                existing_transcripts = {f.replace('.txt', ''): os.path.join("transcripts", f) 
                                     for f in os.listdir("transcripts") 
                                     if f.endswith(".txt")}
                
                missing_videos = [vid_id for vid_id in current_video_ids 
                                if vid_id not in existing_transcripts]
                
                if not missing_videos:
                    # We have all needed transcripts
                    st.session_state.all_files = [existing_transcripts[vid_id] 
                                                for vid_id in current_video_ids 
                                                if vid_id in existing_transcripts]
                    status_container.success(f"Using {len(st.session_state.all_files)} existing transcripts for the selected date range.")
                    
                    with st.spinner('The agent is analyzing the videos... This may take a moment.'):
                        st.session_state.crew = create_agents_and_tasks()
                        st.session_state.response = st.session_state.crew.kickoff(
                            inputs={"file_paths": ", ".join(st.session_state.all_files)})
                    return
                else:
                    status_container.info(f"Found {len(missing_videos)} new videos. Downloading their transcripts...")
            
            # Continue with the rest of the function to display videos and process transcripts
            # ... rest of the existing display and transcript processing code ...

        except Exception as e:
            status_container.error(f"An error occurred: {str(e)}")

# ===========================
#   Sidebar
# ===========================
with st.sidebar:
    st.header("YouTube Channels")

    # Initialize the channels list in session state if it doesn't exist
    if "youtube_channels" not in st.session_state:
        st.session_state.youtube_channels = [""]  # Start with one empty field

    # Function to add new channel field
    def add_channel_field():
        st.session_state.youtube_channels.append("")

    # Create input fields for each channel
    for i, channel in enumerate(st.session_state.youtube_channels):
        col1, col2 = st.columns([6, 1])
        with col1:
            st.session_state.youtube_channels[i] = st.text_input(
                "Channel URL",
                value=channel,
                key=f"channel_{i}",
                label_visibility="collapsed"
            )
        # Show remove button for all except the first field
        with col2:
            if i > 0:
                if st.button("❌", key=f"remove_{i}"):
                    st.session_state.youtube_channels.pop(i)
                    st.rerun()

    # Add channel button
    st.button("Add Channel ➕", on_click=add_channel_field)

    st.divider()

    st.subheader("Date Range")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date")
        st.session_state.start_date = start_date
        # store date as string
        st.session_state.start_date = start_date.strftime("%Y-%m-%d")
    with col2:
        end_date = st.date_input("End Date")
        st.session_state.end_date = end_date
        st.session_state.end_date = end_date.strftime("%Y-%m-%d")

    st.divider()
    
    # Add option to skip transcript download with session state
    st.session_state.skip_transcripts = st.checkbox(
        "Skip transcript download if available", 
        value=st.session_state.skip_transcripts,
        help="If checked, will use existing transcripts instead of downloading new ones"
    )
    
    st.button("Start Analysis 🚀", type="primary", on_click=start_analysis)
    # st.button("Clear Chat", on_click=reset_chat)

# ===========================
#   Main Chat Interface
# ===========================

# Main content area
if st.session_state.response:
    with st.spinner('Generating content... This may take a moment.'):
        try:
            result = st.session_state.response
            st.markdown("### Generated Analysis")
            st.markdown(result)

            # Add download button
            st.download_button(
                label="Download Content",
                data=result.raw,
                file_name=f"youtube_trend_analysis.md",
                mime="text/markdown"
            )
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Built with CrewAI, Bright Data and Streamlit")
