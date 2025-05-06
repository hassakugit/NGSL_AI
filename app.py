# --- app.py ---
import os
import logging
from pathlib import Path
import configparser # Import configparser

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
# from dotenv import load_dotenv # REMOVED
from huggingface_hub import login
from typing import List, Optional # Added Optional for consistency

from translator import Translator # Import the translator class

# --- Configuration Loading ---
# Use configparser to read config.ini
config = configparser.ConfigParser()
config_path = Path('.') / 'config.ini'

# Check if config.ini exists
if not config_path.is_file():
    # Fallback or error handling if config.ini is missing
    # Option 1: Log error and use defaults (potentially risky if models aren't specified)
    # logging.error("config.ini not found! Using hardcoded defaults.")
    # Option 2: Raise an exception to prevent startup without config
    raise FileNotFoundError(f"Configuration file not found: {config_path.resolve()}")

try:
    config.read(config_path)
except configparser.Error as e:
    raise ValueError(f"Error parsing configuration file {config_path.resolve()}: {e}")


# Helper function to get config values with fallbacks
def get_config_value(section, key, fallback=None, type_converter=str):
    try:
        value = config.get(section, key)
        return type_converter(value)
    except (configparser.NoSectionError, configparser.NoOptionError):
        if fallback is not None:
            logging.warning(f"Config '{key}' not found in section '[{section}]'. Using fallback: {fallback}")
            return fallback
        else:
            raise ValueError(f"Required configuration key '{key}' not found in section '[{section}]' and no fallback provided.")
    except ValueError:
        raise ValueError(f"Configuration key '{key}' in section '[{section}]' has an invalid type.")

# Read configuration using the helper function
# Model Names
DEFAULT_TRANSLATION_MODEL = get_config_value('Models', 'TranslationModelName', 'Helsinki-NLP/opus-mt-ja-en')
DEFAULT_CAUSAL_LM_MODEL = get_config_value('Models', 'CausalLMModelName', 'microsoft/Phi-3-mini-128k-instruct')
TRUST_REMOTE_CODE = get_config_value('Models', 'TrustRemoteCode', 'true', type_converter=lambda x: x.lower() == 'true')

# HF Token (optional) - check if section/option exists
HF_HUB_TOKEN = config.get('Credentials', 'HuggingFaceHubToken', fallback=None)
if HF_HUB_TOKEN == "": # Treat empty string as None
    HF_HUB_TOKEN = None

# Device Selection
DEVICE_OVERRIDE = get_config_value('Processing', 'Device', 'auto')

# UI Defaults
DEFAULT_WEEK = get_config_value('UI_Defaults', 'DefaultWeek', 1, type_converter=int)
DEFAULT_THRESHOLD = get_config_value('UI_Defaults', 'DefaultThreshold', 90.0, type_converter=float)

# Advanced
MAX_REWRITE_ATTEMPTS = get_config_value('Processing', 'MaxRewriteAttempts', 5, type_converter=int)


# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Variables ---
translator_instance: Optional[Translator] = None # Use Optional
selected_device: Optional[str] = None

# --- Helper Functions ---
def determine_device() -> str:
    """Determines the optimal device based on availability and override."""
    if DEVICE_OVERRIDE and DEVICE_OVERRIDE.lower() in ["cuda", "mps", "cpu"]:
        device = DEVICE_OVERRIDE.lower()
        logger.info(f"Using device specified in config.ini: {device}")
        # Verify availability if specific device requested
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA specified but not available. Falling back.")
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        elif device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS specified but not available. Falling back to CPU.")
            device = "cpu"
        return device
    else:
        # Auto-detection order: CUDA > MPS > CPU
        if torch.cuda.is_available():
            logger.info("CUDA detected, using CUDA.")
            return "cuda"
        elif torch.backends.mps.is_available():
            try:
                torch.zeros(1).to('mps')
                logger.info("MPS detected, using MPS.")
                return "mps"
            except Exception as e:
                 logger.warning(f"MPS device detected but failed usability test ({e}). Falling back to CPU.")
                 return "cpu"
        else:
            logger.info("No GPU (CUDA/MPS) detected, using CPU.")
            return "cpu"

# --- FastAPI Application Setup ---
app = FastAPI(title="NGSL Translator", description="Translate Japanese to English with NGSL vocabulary constraints.")

# --- Startup Event Handler ---
@app.on_event("startup")
async def startup_event():
    """Code to run when the application starts."""
    global translator_instance, selected_device
    logger.info("Application startup sequence initiated...")

    # 1. Determine Device
    selected_device = determine_device()
    logger.info(f"Selected device for models: {selected_device}")

    # 2. Authenticate with Hugging Face Hub (if token provided)
    if HF_HUB_TOKEN:
        logger.info("HuggingFaceHubToken found in config.ini. Attempting Hugging Face Hub login...")
        try:
            login(token=HF_HUB_TOKEN)
            logger.info("Hugging Face Hub login successful.")
        except Exception as e:
            logger.error(f"Hugging Face Hub login failed: {e}", exc_info=True)
            logger.warning("Proceeding without Hub authentication. Model download might fail for private/gated models.")
    else:
        logger.info("No HuggingFaceHubToken provided in config.ini. Skipping Hugging Face Hub login.")

    # 3. Pre-load models and initialize translator
    logger.info("Initializing Translator instance (this may take time for model downloads)...")
    try:
        vocab_path = Path("./vocablists")
        translator_instance = Translator(
            translation_model_name=DEFAULT_TRANSLATION_MODEL,
            causal_lm_model_name=DEFAULT_CAUSAL_LM_MODEL,
            device=selected_device,
            vocab_dir=str(vocab_path),
            trust_remote_code=TRUST_REMOTE_CODE,
            max_rewrite_attempts=MAX_REWRITE_ATTEMPTS
        )
        logger.info("Translator initialized successfully.")
    except Exception as e:
        logger.error(f"FATAL: Failed to initialize Translator during startup: {e}", exc_info=True)
        translator_instance = None

    logger.info("Application startup sequence complete.")


# --- Pydantic Models for API ---
class TranslateRequest(BaseModel):
    japanese: str = Field(..., example="これは複雑な日本語の文章です。")
    week: int = Field(..., ge=1, example=5)
    threshold: float = Field(..., ge=0, le=100, example=90.0)

class TranslateResponse(BaseModel):
    original_japanese: str
    simplified_japanese: List[str]
    initial_english: str
    final_english: str
    coverage: float
    violations: List[str]

class ConfigResponse(BaseModel):
    translation_model: str
    causal_lm_model: str
    device: str
    default_week: int
    default_threshold: float
    available_weeks: List[int]


# --- API Endpoints ---

# Serve static frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serves the main index.html file."""
    html_path = Path("frontend/index.html")
    if not html_path.is_file():
        raise HTTPException(status_code=404, detail="index.html not found")
    return HTMLResponse(content=html_path.read_text(), status_code=200)

@app.get("/api/config", response_model=ConfigResponse)
async def get_config():
    """Returns the current backend configuration and UI defaults."""
    if translator_instance is None:
         raise HTTPException(status_code=503, detail="Translator service is not available (initialization failed).")
    if selected_device is None: # Should not happen if startup completed but good check
        raise HTTPException(status_code=503, detail="Device configuration not determined.")


    available_weeks = sorted(translator_instance.vocab_lists.keys()) if translator_instance and translator_instance.vocab_lists else []

    return ConfigResponse(
        translation_model=DEFAULT_TRANSLATION_MODEL,
        causal_lm_model=DEFAULT_CAUSAL_LM_MODEL,
        device=selected_device,
        default_week=DEFAULT_WEEK,
        default_threshold=DEFAULT_THRESHOLD,
        available_weeks=available_weeks
    )

@app.post("/api/translate", response_model=TranslateResponse)
async def translate_text(request: TranslateRequest):
    """Receives Japanese text, translates it, and applies vocabulary constraints."""
    if translator_instance is None:
        raise HTTPException(status_code=503, detail="Translator service is not available (initialization failed).")

    if not request.japanese:
        raise HTTPException(status_code=400, detail="Input 'japanese' text cannot be empty.")

    if request.week not in translator_instance.vocab_lists:
         logger.warning(f"Requested week {request.week} vocabulary list not found. Proceeding without vocab check.")
         # Optionally raise: raise HTTPException(status_code=400, detail=f"Vocabulary list for week {request.week} not found.")


    try:
        logger.info(f"Processing translation request: week={request.week}, threshold={request.threshold}")
        result = translator_instance.translate(
            japanese_text=request.japanese,
            week=request.week,
            threshold=request.threshold
        )
        return result
    except Exception as e:
        logger.error(f"Error during translation processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred during translation: {e}")
