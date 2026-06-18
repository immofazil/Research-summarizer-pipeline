import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List

# Load configuration
load_dotenv()

# Initialize the Gemini Client
client = genai.Client()
MODEL_ID = "gemini-2.5-flash"

# --- Embedded Pydantic Schema ---
class KeyPointsExtractor(BaseModel):
    key_points: List[str] = Field(
        description="A list of core findings, key facts, or critical insights extracted from the text."
    )

# --- Core Pipeline Functions ---
def extract_key_points(text: str) -> KeyPointsExtractor:
    """Step 1: Extract core takeaways into a strict structured JSON list."""
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
    # Validate and convert directly into the Pydantic structure
    return KeyPointsExtractor.model_validate_json(response.text)

def generate_research_brief(extracted_points: KeyPointsExtractor) -> str:
    """Step 2: Take the structured JSON list and turn it into an executive brief."""
    # Convert the structured list back into a clean string block for the next call
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
    return response.text.strip()

def run_pipeline(input_text: str) -> dict:
    """Orchestrates the data extraction and briefing chain."""
    print("Executing Step 1: Extracting Structured Key Points...")
    extracted_data = extract_key_points(input_text)
    
    print("Executing Step 2: Chaining Data to Generate Research Brief...")
    final_brief = generate_research_brief(extracted_data)
    
    # Bundle both the intermediate payload and final result together
    return {
        "intermediate_json": extracted_data.model_dump(),
        "final_brief": final_brief
    }

if __name__ == "__main__":
    # Test Payload
    sample_research = (
        "Recent breakthroughs in Model Context Protocol (MCP) architectures have significantly reduced "
        "latency in multi-agent systems. By establishing an open standard for how AI tools connect to data sources, "
        "enterprise implementations saw a 40% reduction in custom middleware code. However, current challenges "
        "include handling malformed JSON from older legacy models and establishing secure authentication pathways "
        "across public endpoints. Production deployments are expected to double over the next fiscal year."
    )
    
    print("Starting Research Summarizer Pipeline...")
    print(f"Input Text:\n{sample_research}\n" + "-"*50)
    
    try:
        pipeline_output = run_pipeline(sample_research)
        print("\n" + "="*50 + "\nFinal Pipeline Output:")
        print(json.dumps(pipeline_output, indent=2))
    except Exception as e:
        print(f"\nPipeline Error Encountered: {e}")