import os
import google.generativeai as genai
from loguru import logger
from typing import Optional, Dict, Any

class GeminiService:
    """Service class for Google Gemini API interactions"""
    
    def __init__(self):
        """Initialize Gemini client with API key from environment"""
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found in environment variables")
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.configured = True
        else:
            self.configured = False
    
    def chat_completion(self, prompt: str, model: str = "gemini-2.0-flash-lite",
                       max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """
        Generate chat completion using Google Gemini API
        
        Args:
            prompt: The input prompt
            model: Gemini model to use
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            
        Returns:
            Generated response text
        """
        if not self.configured:
            raise Exception("Gemini API key not configured")
        
        try:
            # Initialize the model
            model_instance = genai.GenerativeModel(model)
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
            
            # Generate response
            response = model_instance.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            if response.text:
                return response.text.strip()
            else:
                raise Exception("No response content received from Gemini API")
                
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            raise Exception(f"Gemini API error: {str(e)}")
    
    def chat_with_history(self, prompt: str, history: list = None,
                         model: str = "gemini-2.0-flash-lite", max_tokens: int = 1000,
                         temperature: float = 0.7) -> str:
        """
        Generate chat completion with conversation history using Gemini API
        
        Args:
            prompt: The current input prompt
            history: List of previous messages
            model: Gemini model to use
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            
        Returns:
            Generated response text
        """
        if not self.configured:
            raise Exception("Gemini API key not configured")
        
        try:
            # Initialize the model
            model_instance = genai.GenerativeModel(model)
            
            # Start chat with history
            chat = model_instance.start_chat(history=history or [])
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
            
            # Send message and get response
            response = chat.send_message(
                prompt,
                generation_config=generation_config
            )
            
            if response.text:
                return response.text.strip()
            else:
                raise Exception("No response content received from Gemini API")
                
        except Exception as e:
            logger.error(f"Gemini Chat API error: {str(e)}")
            raise Exception(f"Gemini Chat API error: {str(e)}")
    
    def generate_with_image(self, prompt: str, image_path: str = None,
                           model: str = "gemini-2.0-flash-lite-vision", max_tokens: int = 1000,
                           temperature: float = 0.7) -> str:
        """
        Generate content with image input using Gemini Vision API
        
        Args:
            prompt: The text prompt
            image_path: Path to the image file
            model: Gemini vision model to use
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            
        Returns:
            Generated response text
        """
        if not self.configured:
            raise Exception("Gemini API key not configured")
        
        if not image_path or not os.path.exists(image_path):
            raise Exception("Valid image path is required for vision model")
        
        try:
            # Initialize the vision model
            model_instance = genai.GenerativeModel(model)
            
            # Load and prepare image
            import PIL.Image
            image = PIL.Image.open(image_path)
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
            
            # Generate response with image
            response = model_instance.generate_content(
                [prompt, image],
                generation_config=generation_config
            )
            
            if response.text:
                return response.text.strip()
            else:
                raise Exception("No response content received from Gemini Vision API")
                
        except Exception as e:
            logger.error(f"Gemini Vision API error: {str(e)}")
            raise Exception(f"Gemini Vision API error: {str(e)}")
    
    def stream_completion(self, prompt: str, model: str = "gemini-2.0-flash-lite",
                         max_tokens: int = 1000, temperature: float = 0.7):
        """
        Generate streaming chat completion using Gemini API
        
        Args:
            prompt: The input prompt
            model: Gemini model to use
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            
        Yields:
            Streaming response chunks
        """
        if not self.configured:
            raise Exception("Gemini API key not configured")
        
        try:
            # Initialize the model
            model_instance = genai.GenerativeModel(model)
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
            
            # Generate streaming response
            response = model_instance.generate_content(
                prompt,
                generation_config=generation_config,
                stream=True
            )
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            logger.error(f"Gemini Streaming API error: {str(e)}")
            raise Exception(f"Gemini Streaming API error: {str(e)}")
    
    def is_configured(self) -> bool:
        """Check if Gemini service is properly configured"""
        return self.configured 