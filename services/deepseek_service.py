import os
import requests
from loguru import logger
from typing import Optional, Dict, Any

class DeepSeekService:
    """Service class for DeepSeek API interactions"""
    
    def __init__(self):
        """Initialize DeepSeek client with API key from environment"""
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        self.base_url = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1')
        
        if not self.api_key:
            logger.warning("DEEPSEEK_API_KEY not found in environment variables")
        
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        } if self.api_key else None
    
    def chat_completion(self, prompt: str, model: str = "deepseek-chat",
                       max_tokens: int = 7000, temperature: float = 0.7) -> str:
        """
        Generate chat completion using DeepSeek API
        
        Args:
            prompt: The input prompt
            model: DeepSeek model to use
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            
        Returns:
            Generated response text
        """
        if not self.headers:
            raise Exception("DeepSeek API key not configured")
        
        try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content'].strip()
            else:
                raise Exception("No response content received from DeepSeek API")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"DeepSeek API request error: {str(e)}")
            raise Exception(f"DeepSeek API request error: {str(e)}")
        except Exception as e:
            logger.error(f"DeepSeek API error: {str(e)}")
            raise Exception(f"DeepSeek API error: {str(e)}")
    
    def code_completion(self, prompt: str, model: str = "deepseek-coder",
                       max_tokens: int = 1000, temperature: float = 0.1) -> str:
        """
        Generate code completion using DeepSeek Coder API
        
        Args:
            prompt: The input code prompt
            model: DeepSeek coder model to use
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (lower for code)
            
        Returns:
            Generated code text
        """
        if not self.headers:
            raise Exception("DeepSeek API key not configured")
        
        try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content'].strip()
            else:
                raise Exception("No response content received from DeepSeek API")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"DeepSeek Coder API request error: {str(e)}")
            raise Exception(f"DeepSeek Coder API request error: {str(e)}")
        except Exception as e:
            logger.error(f"DeepSeek Coder API error: {str(e)}")
            raise Exception(f"DeepSeek Coder API error: {str(e)}")
    
    def stream_completion(self, prompt: str, model: str = "deepseek-chat",
                         max_tokens: int = 1000, temperature: float = 0.7):
        """
        Generate streaming chat completion using DeepSeek API
        
        Args:
            prompt: The input prompt
            model: DeepSeek model to use
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            
        Yields:
            Streaming response chunks
        """
        if not self.headers:
            raise Exception("DeepSeek API key not configured")
        
        try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                stream=True,
                timeout=30
            )
            
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]
                        if data != '[DONE]':
                            try:
                                import json
                                chunk = json.loads(data)
                                if 'choices' in chunk and len(chunk['choices']) > 0:
                                    delta = chunk['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        yield delta['content']
                            except json.JSONDecodeError:
                                continue
                                
        except requests.exceptions.RequestException as e:
            logger.error(f"DeepSeek Streaming API request error: {str(e)}")
            raise Exception(f"DeepSeek Streaming API request error: {str(e)}")
        except Exception as e:
            logger.error(f"DeepSeek Streaming API error: {str(e)}")
            raise Exception(f"DeepSeek Streaming API error: {str(e)}")
    
    def is_configured(self) -> bool:
        """Check if DeepSeek service is properly configured"""
        return self.api_key is not None 