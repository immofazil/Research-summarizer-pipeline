import os
import json
import time
import logging
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError
from typing import List

# ==========================================
# 1. Configuration & Structured Logging
# ==========================================
# Set up logging to track timestamps, step names, and status
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Load environment variables (NEVER hardcode secrets)
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    logging.critical("GEMINI_API_KEY is missing from the .env file.")
    raise ValueError("Missing API Key. Please check your .env file.")

# Initialize the client securely
client = genai.Client()
MODEL_ID = "gemini-2.5-flash"

# ==========================================
# 2. Pydantic Schema
# ==========================================
class KeyPointsExtractor(BaseModel):
    key_points: List[str] = Field(
        description="A list of core findings, key facts, or critical insights extracted from the text."
    )

# ==========================================
# 3. Retry Logic (Exponential Backoff)
# ==========================================
def with_exponential_backoff(max_retries=3, base_delay=2):
    """Decorator to retry API calls if they fail, doubling the wait time each attempt."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    logging.warning(f"Attempt {attempt} failed: {str(e)}")
                    if attempt >= max_retries:
                        logging.error(f"Max retries reached. Operation failed.")
                        raise e
                    
                    sleep_time = base_delay * (2 ** (attempt - 1))
                    logging.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
        return wrapper
    return decorator

# ==========================================
# 4. Pipeline Steps
# ==========================================
@with_exponential_backoff(max_retries=3, base_delay=2)
def extract_key_points(text: str) -> KeyPointsExtractor:
    """Step 1: Extract core takeaways into a strict structured JSON list."""
    logging.info("Starting Step 1: Extracting key points to JSON.")
    
    prompt = f"Analyze the following text and extract all core findings, key facts, and data points:\n\n{text}"
    
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=KeyPointsExtractor,
            temperature=0.1
        ),
    )
    
    try:
        # Parse and validate the JSON directly into our Pydantic structure
        parsed_data = KeyPointsExtractor.model_validate_json(response.text)
        logging.info("Step 1 Complete: Successfully parsed JSON.")
        return parsed_data
    except ValidationError as e:
        logging.error("Malformed JSON returned by the model.")
        raise ValueError(f"Failed to parse structured output: {e}")

@with_exponential_backoff(max_retries=3, base_delay=2)
def generate_research_brief(extracted_points: KeyPointsExtractor) -> str:
    """Step 2: Take the structured JSON list and turn it into an executive brief."""
    logging.info("Starting Step 2: Generating research brief from structured data.")
    
    formatted_points = "\n".join([f"- {point}" for point in extracted_points.key_points])
    
    prompt = (
        f"You are an expert research analyst. Transform these extracted data points into a formal, "
        f"high-level executive brief with clear headings (Overview, Key Takeaways, Next Steps):\n\n{formatted_points}"
    )
    
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.5)
    )
    
    logging.info("Step 2 Complete: Research brief generated.")
    return response.text.strip()

def run_pipeline(input_text: str) -> dict:
    """Orchestrates the entire chaining process with safety checks."""
    logging.info("Pipeline execution started.")
    
    # Error Handling: Check for empty input
    if not input_text or not input_text.strip():
        logging.error("Input validation failed: Empty text provided.")
        raise ValueError("The input text cannot be empty. Please provide content to analyze.")
        
    extracted_data = extract_key_points(input_text)
    final_brief = generate_research_brief(extracted_data)
    
    logging.info("Pipeline execution finished successfully.")
    
    return {
        "intermediate_json": extracted_data.model_dump(),
        "final_brief": final_brief
    }

# ==========================================
# 5. Terminal Execution Entry Point
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("RESEARCH SUMMARIZER PIPELINE (Local Terminal Mode)")
    print("="*60 + "\n")
    
    # Wait for the user to paste their text
    user_input = input("Paste your research text or topic here and press Enter:\n\n> ")
    
    print("\n" + "-"*60)
    
    try:
        # Run the robust pipeline
        pipeline_output = run_pipeline(user_input)
        
        print("\n" + "="*60)
        print("FINAL PIPELINE OUTPUT:")
        print("="*60)
        print(json.dumps(pipeline_output, indent=2))
        
    except Exception as e:
        print("\n" + "!"*60)
        print(f"PIPELINE HALTED: {e}")
        print("!"*60)