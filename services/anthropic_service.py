import os
import anthropic
from loguru import logger
from typing import Optional, Dict, Any

class AnthropicService:
    """Service class for Anthropic Claude API interactions"""
    
    def __init__(self):
        """Initialize Anthropic client with API key from environment"""
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            logger.warning("ANTHROPIC_API_KEY not found in environment variables")
        
        self.client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else None
    
    def chat_completion(self, prompt: str, model: str = "claude-3-haiku-20240307",
                       max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """
        Generate chat completion using Anthropic Claude API
        
        Args:
            prompt: The input prompt
            model: Claude model to use
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            
        Returns:
            Generated response text
        """
        if not self.client:
            raise Exception("Anthropic API key not configured")
        
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise Exception(f"Anthropic API error: {str(e)}")
    
    def chat_with_system(self, prompt: str, system_prompt: str = "",
                        model: str = "claude-3-haiku-20240307",
                        max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """
        Generate chat completion with system prompt using Anthropic Claude API
        
        Args:
            prompt: The user input prompt
            system_prompt: System instruction prompt
            model: Claude model to use
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            
        Returns:
            Generated response text
        """
        if not self.client:
            raise Exception("Anthropic API key not configured")
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise Exception(f"Anthropic API error: {str(e)}")
    
    def stream_completion(self, prompt: str, model: str = "claude-3-haiku-20240307",
                         max_tokens: int = 1000, temperature: float = 0.7):
        """
        Generate streaming chat completion using Anthropic Claude API
        
        Args:
            prompt: The input prompt
            model: Claude model to use
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            
        Yields:
            Streaming response chunks
        """
        if not self.client:
            raise Exception("Anthropic API key not configured")
        
        try:
            with self.client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            ) as stream:
                for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            logger.error(f"Anthropic Streaming API error: {str(e)}")
            raise Exception(f"Anthropic Streaming API error: {str(e)}")
    
    def is_configured(self) -> bool:
        """Check if Anthropic service is properly configured"""
        return self.api_key is not None 