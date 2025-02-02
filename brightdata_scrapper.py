import subprocess
import json
import time
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("BRIGHT_DATA_API_KEY")

def trigger_scraping_niche(api_key, keyword, num_of_posts, start_date, end_date, country, endpoint):

    payload = [{"keyword": keyword, 
                "num_of_posts": num_of_posts, 
                "start_date": start_date, 
                "end_date": end_date, 
                "country": country}]

    # Define the curl command
    command = [
        "curl",
        "-H", f"Authorization: Bearer {api_key}",
        "-H", "Content-Type: application/json",
        "-d", json.dumps(payload),  # Convert payload to a JSON string
        endpoint
    ]

    # Execute the command and capture the output
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # Check if the command was successful
    if result.returncode == 0:
        try:
            # Parse the JSON response
            return json.loads(result.stdout.strip())
        except json.JSONDecodeError:
            print("Failed to parse JSON response.")
            return None
    else:
        # Print the error if the command fails
        print(f"Error: {result.stderr}")
        return None

def trigger_scraping_channels(api_key, channel_urls, num_of_posts, start_date, end_date, order_by, country):
    
    # Validate num_of_posts
    if num_of_posts > 50:
        return {
            "status": "failed",
            "error": "Number of posts cannot exceed 50 per request. Please use date ranges for more videos."
        }
    
    dataset_id = "gd_lk56epmy2i5g7lzu0k"
    endpoint = f"https://api.brightdata.com/datasets/v3/trigger?dataset_id={dataset_id}&include_errors=true&type=discover_new&discover_by=url"

    payload = [
        {
            "url": url,
            "num_of_posts": num_of_posts,
            "start_date": start_date,
            "end_date": end_date,
            "order_by": order_by,
            "country": country
        }
        for url in channel_urls
    ]

    # Define the curl command
    command = [
        "curl",
        "-H", f"Authorization: Bearer {api_key}",
        "-H", "Content-Type: application/json",
        "-d", json.dumps(payload),
        endpoint
    ]

    # Execute the command and capture the output
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Check if the command was successful
    if result.returncode == 0:
        try:
            response = json.loads(result.stdout.strip())
            
            # Check for specific error messages in the response
            if isinstance(response, dict) and response.get('error'):
                error_msg = response.get('error')
                if 'limit' in error_msg.lower():
                    return {
                        "status": "failed",
                        "error": "Video limit exceeded. Maximum 50 videos per request. Please use date ranges for more videos."
                    }
                return {"status": "failed", "error": error_msg}
            
            return response
            
        except json.JSONDecodeError:
            print("Failed to parse JSON response.")
            return {"status": "failed", "error": "Failed to parse API response"}
    else:
        # Print and return the error if the command fails
        error_msg = result.stderr
        print(f"Error: {error_msg}")
        return {"status": "failed", "error": error_msg}

def get_progress(api_key, snapshot_id):
    command = [
        "curl",
        "-H", f"Authorization: Bearer {api_key}",
        f"https://api.brightdata.com/datasets/v3/progress/{snapshot_id}"
    ]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode == 0:
        return json.loads(result.stdout.strip())
    else:
        print(f"Error: {result.stderr}")
        return None

def get_output(api_key, snapshot_id, format="json"):

    # Define the curl command as a list
    command = [
        "curl",
        "-H", f"Authorization: Bearer {api_key}",
        f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}?format={format}"
    ]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode == 0:
        json_lines = result.stdout.strip().split("\n")
        print(json_lines)
        json_objects = [json.loads(line) for line in json_lines]
        return json_objects
    else:
        print(f"Error: {result.stderr}")



