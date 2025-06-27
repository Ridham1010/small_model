import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from fastapi.middleware.cors import CORSMiddleware

# --- For Local Development Only: Load environment variables from .env file ---
# IMPORTANT: Make sure you have `pip install python-dotenv` if running locally.
# This line should NOT be present in your production deployment script on Render,
# as Render handles environment variables directly.
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("'.env' file loaded for local development.")
except ImportError:
    print("python-dotenv not found. If running locally, consider installing it to load .env variables.")


# --- Configuration ---
model_name = "google/gemma-2b-it"

# Read the Hugging Face token from environment variables
# This will pick up from Render's environment variables or your local .env file
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    # It's crucial to handle cases where the token is not set
    print("ERROR: HF_TOKEN environment variable not set. Model loading might fail.")
    # In a real application, you might want to raise an exception or provide a fallback
    # raise ValueError("Hugging Face token (HF_TOKEN) is not configured in environment variables.")

# --- Global Variables for Model ---
# These will be loaded once when the FastAPI app starts
llm_pipeline = None
tokenizer = None

# --- Lifespan Context Manager ---
# This function will handle startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup events (equivalent to @app.on_event("startup"))
    global llm_pipeline, tokenizer

    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN) # Pass token here
    print("Tokenizer loaded.")

    print(f"Loading model for {model_name}...")
    # Determine device for PyTorch
    if torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = torch.float16 # Using float16 for stability and memory efficiency on MPS
        print(f"Using Apple Silicon (MPS) for acceleration with {torch_dtype}.")
    elif torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.float16 # Using float16 for stability and memory efficiency on CUDA
        print(f"Using CUDA (GPU) for acceleration with {torch_dtype}.")
    else:
        device = "cpu"
        torch_dtype = None # Default to float32 on CPU
        print("No GPU detected (MPS/CUDA). Using CPU, which will be slow for LLMs.")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            token=HF_TOKEN # Pass token here
        )
        model.to(device) # Explicitly move model to the determined device

        print("Model loaded.")
        print(f"Model device after explicit move: {model.device}")

        llm_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=50, # Keep it low for initial testing
            temperature=0.7,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.1,
            device=device # Ensure pipeline uses the correct device
        )
        print("\n--- LLM Backend Ready ---")
    except Exception as e:
        print(f"Failed to load model or pipeline: {e}")
        llm_pipeline = None # Indicate failure
        tokenizer = None # Indicate failure
        # Depending on severity, you might want to exit the application or mark it as unhealthy

    yield # This yields control to the FastAPI application

    # Shutdown events (equivalent to @app.on_event("shutdown"))
    print("Shutting down LLM backend...")
    # If using GPU, clear cache to free memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        # MPS doesn't have a direct empty_cache equivalent in the same way as CUDA
        # but releasing references can help if objects are GC'd
        pass


# --- FastAPI App Initialization ---
# Pass the lifespan context manager to the FastAPI instance
app = FastAPI(lifespan=lifespan)

# --- CORS Middleware (Crucial for frontend interaction) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development. Be more specific in production!
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# --- Pydantic Model for Request Body ---
class PromptRequest(BaseModel):
    prompt: str

# --- API Endpoint for LLM Inference ---
@app.post("/generate")
async def generate_text(request: PromptRequest):
    user_input = request.prompt

    # Ensure llm_pipeline and tokenizer are loaded (should be by lifespan)
    if llm_pipeline is None or tokenizer is None:
        return {"error": "Model not loaded yet. Please wait or restart the server. Check backend logs for errors."}, 503

    # Format input for instruct models (same logic as run_llm.py)
    if "gemma" in model_name.lower():
        messages = [{"role": "user", "content": user_input}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif "mistral" in model_name.lower():
        prompt = f"<s>[INST] {user_input} [/INST]"
    else:
        prompt = user_input

    try:
        response = llm_pipeline(prompt)
        generated_text = response[0]['generated_text']

        # Post-processing to remove the prompt from the generated text
        if "gemma" in model_name.lower() or "mistral" in model_name.lower():
            # Adjust marker based on model's exact response format
            # For Gemma, it often starts with "<start_of_turn>model\n" or similar after the prompt
            # For Mistral, the prompt ends with "[/INST]"
            if "gemma" in model_name.lower():
                # Attempt to find common chat template markers or the end of the input prompt
                # The exact marker can vary slightly, so try both or more robust parsing
                start_marker_1 = "<start_of_turn>model\n"
                start_marker_2 = "<bos>" + prompt # Sometimes the whole prompt is prefixed with <bos>

                start_of_response_idx = -1
                if start_marker_1 in generated_text:
                    start_of_response_idx = generated_text.rfind(start_marker_1)
                elif generated_text.startswith(prompt):
                    start_of_response_idx = generated_text.find(prompt) + len(prompt)

                if start_of_response_idx != -1:
                    generated_text = generated_text[start_of_response_idx:].strip()
                    # Further trim any potential remaining user prompt if it was duplicated
                    if generated_text.startswith(user_input):
                        generated_text = generated_text[len(user_input):].strip()

            elif "mistral" in model_name.lower():
                start_marker = "[/INST]"
                start_of_response_idx = generated_text.rfind(start_marker)
                if start_of_response_idx != -1:
                    generated_text = generated_text[start_of_response_idx + len(start_marker):].strip()
                else:
                    if generated_text.startswith(prompt):
                        generated_text = generated_text[len(prompt):].strip()
            else:
                # For non-instruct models, just remove the initial prompt if it's there
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()

        return {"response": generated_text.strip()}

    except Exception as e:
        # Log the full exception for debugging, but send a generic error to the frontend
        print(f"Error during text generation: {e}")
        return {"error": "An error occurred during text generation. Check server logs."}, 500


# Optional: Basic health check endpoint
@app.get("/")
async def root():
    return {"message": "LLM API is running!"}