import os
import openai
from loguru import logger
from typing import Optional, Dict, Any

class OpenAIService:
    """Service class for OpenAI API interactions"""
    
    def __init__(self):
        """Initialize OpenAI client with API key from environment"""
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables")
        
        openai.api_key = self.api_key
        self.client = openai.OpenAI(api_key=self.api_key) if self.api_key else None
    
    def chat_completion(self, prompt: str, model: str = "gpt-3.5-turbo", 
                       max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """
        Generate chat completion using OpenAI API
        
        Args:
            prompt: The input prompt
            model: OpenAI model to use
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            
        Returns:
            Generated response text
        """
        if not self.client:
            raise Exception("OpenAI API key not configured")
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def text_completion(self, prompt: str, model: str = "gpt-3.5-turbo-instruct",
                       max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """
        Generate text completion using OpenAI API
        
        Args:
            prompt: The input prompt
            model: OpenAI model to use
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            
        Returns:
            Generated response text
        """
        if not self.client:
            raise Exception("OpenAI API key not configured")
        
        try:
            response = self.client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].text.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def get_embeddings(self, text: str, model: str = "text-embedding-ada-002") -> list:
        """
        Get text embeddings using OpenAI API
        
        Args:
            text: Input text to embed
            model: Embedding model to use
            
        Returns:
            List of embedding values
        """
        if not self.client:
            raise Exception("OpenAI API key not configured")
        
        try:
            response = self.client.embeddings.create(
                model=model,
                input=text
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"OpenAI Embeddings API error: {str(e)}")
            raise Exception(f"OpenAI Embeddings API error: {str(e)}")
    
    def is_configured(self) -> bool:
        """Check if OpenAI service is properly configured"""
        return self.api_key is not None 