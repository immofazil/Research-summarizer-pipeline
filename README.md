Here is the updated, structured setup guide for your `README.txt` file, complete with the direct link to obtain your Gemini API key.

```text
========================================================================
RESEARCH SUMMARIZER PIPELINE
========================================================================

A production-ready Python pipeline that demonstrates structured chaining 
using an LLM. The system takes a long research text or topic, extracts 
key points into a strict JSON list format, and feeds that structured 
data directly into a second step to generate a formal research brief.


PROJECT ARCHITECTURE
------------------------------------------------------------------------
[Raw Text Input] 
       │
       ▼
[Step 1: Extract Key Points] ──► Generates Strict Pydantic JSON Schema
       │
       ▼
[Step 2: Generate Research Brief] ──► Consumes Intermediate JSON
       │
       ▼
[Final Unified JSON Output] ──► Contains both Brief & JSON List


CORE FEATURES
------------------------------------------------------------------------
* Data Chaining: Implements linear chaining where the structured JSON 
  output of the extraction step serves as the exact input for the 
  summarization step.
* Structured Output: Uses Pydantic to enforce an exact JSON schema for 
  data extraction, minimizing parsing errors.
* Intermediate State Retention: Returns both the raw extracted bullet 
  points and the final research brief in a unified payload.


SETUP INSTRUCTIONS
------------------------------------------------------------------------

1. Navigate to the project root directory:
   cd research-summarizer-pipeline

2. Create and activate an isolated Python virtual environment:
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate

3. Install all required software dependencies:
   pip install -r requirements.txt

4. Configure your local environment variables:
   - Get an API key from Google AI Studio: https://aistudio.google.com/
   - Create a file named ".env" in the root directory.
   - Add your credentials to the file exactly as shown below:
     
     GEMINI_API_KEY=your_actual_api_key_here

5. Execute the pipeline script:
   python -m src.pipeline

========================================================================

```