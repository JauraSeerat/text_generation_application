"""
This module implements a text generation API using FastAPI and Hugging Face Transformers
It provides two main endpoints:
  - GET "/" returns the index page rendered using a Jinja2 template
  - POST "/generate" accepts a prompt with optional hyperparameters and returns generated text

"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from pydantic import BaseModel
from transformers import pipeline, set_seed, logging as transformers_logging
import logging
import os

# Disable CPU features check and suppress unnecessary warnings from the transformers library.
os.environ["TORCH_CPU_FEATURES_CHECK"] = "0"
transformers_logging.set_verbosity_error()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI application and Jinja2 template renderer.
app = FastAPI(title="Text Generation Tool")
templates = Jinja2Templates(directory="templates")


class PromptRequest(BaseModel):
    """
    Data model for a text generation

    Attributes:
        prompt (str): The input text prompt for generation.
        max_length (int): The maximum number of additional tokens needed to generate (default is 150)
        temperature (float): The sampling temperature to control randomness (default is 0.7)
        top_p (float): The nucleus sampling probability threshold (default is 0.95)
    """
    prompt: str
    max_length: int = 150
    temperature: float = 0.7
    top_p: float = 0.95


class TextGenerator:
    """
    A wrapper class for generating text using a pretrained language model via Hugging Face's pipeline
    
    Attributes:
        generator: The instance of Hugging Face text-generation pipeline 
        tokenizer: The tokenizer associated with the language model
    """
    
    def __init__(self, model_name: str = "gpt2", seed: int = 42):
        """
        Initialize the TextGenerator with a specific model and random seed

        Args:
            model_name (str, optional): The name of the pretrained model to use (default in this case "gpt2")
            seed (int, optional): The seed for random number generation to ensure reproducibility (default is 42).
        """
        logger.info("Running text generation model...")
        self.generator = pipeline("text-generation", model=model_name)
        self.tokenizer = self.generator.tokenizer
        # Ensure the tokenizer's padding token is set to the end-of-sequence token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        set_seed(seed)

    def generate(self, prompt: str, max_length: int, temperature: float, top_p: float) -> str:
        """
        Generate text based on the provided prompt and generation parameters

        Args:
            prompt (str): The input prompt for text generation
            max_length (int): The maximum additional token length to generate
            temperature (float): The sampling temperature (controls randomness)
            top_p (float): The nucleus sampling probability threshold

        Returns:
            str: The generated text.

        Raises:
            HTTPException: If text generation fails
        """
        try:
            # Tokenize the prompt to calculate the adjusted maximum length
            prompt_tokens = self.tokenizer.tokenize(prompt)
            # Ensure the total length does not exceed the model's maximum allowed tokens
            adjusted_max_length = min(len(prompt_tokens) + max_length, self.tokenizer.model_max_length)

            result = self.generator(
                prompt,
                max_length=adjusted_max_length,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                truncation=True
            )
            return result[0]["generated_text"]
        except Exception as e:
            logger.error(f"Text generation error: {e}")
            raise HTTPException(status_code=500, detail="Text generation failed.")


@app.get("/", response_class=HTMLResponse)
def read_index(request: Request):
    """
    Render and return the index HTML page.

    Args:
        request (Request): The incoming HTTP request.

    Returns:
        HTMLResponse: The rendered 'index.html' template with the provided request context.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/generate")
async def generate_text(promptrequest: PromptRequest):
    """
    Generate text using the provided prompt and optional hyperparameters.

    Args:
        promptrequest (PromptRequest): The request body containing the prompt and generation parameters.

    Returns:
        dict: A dictionary with the key 'generated_text' containing the generated text.

    Raises:
        HTTPException: If the prompt is empty or text generation fails.
    """
    if not promptrequest.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    generator = TextGenerator()
    generated_text = generator.generate(promptrequest.prompt, promptrequest.max_length, promptrequest.temperature, promptrequest.top_p)
    return {"generated_text": generated_text}
