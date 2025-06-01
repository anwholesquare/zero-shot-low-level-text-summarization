"""
Configuration settings for AI Services Flask API
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Base configuration class"""
    
    # Flask Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    PORT = int(os.getenv('FLASK_PORT', 5000))
    
    # API Keys
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    
    # Service URLs
    DEEPSEEK_BASE_URL = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1')
    
    # Model Configurations
    DEFAULT_MODELS = {
        'openai': 'gpt-3.5-turbo',
        'anthropic': 'claude-3-haiku-20240307',
        'deepseek': 'deepseek-chat',
        'gemini': 'gemini-pro',
        'bloomz': 'bigscience/bloomz-560m'
    }
    
    # Generation Parameters
    DEFAULT_MAX_TOKENS = 1000
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_TOP_P = 0.9
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = 'logs/app.log'
    LOG_ROTATION = '1 day'
    LOG_RETENTION = '7 days'
    
    # Rate Limiting (requests per minute)
    RATE_LIMITS = {
        'openai': 60,
        'anthropic': 60,
        'deepseek': 60,
        'gemini': 60,
        'bloomz': 30  # Lower for local model
    }
    
    # Bloomz Model Configuration
    BLOOMZ_MODEL_NAME = os.getenv('BLOOMZ_MODEL_NAME', 'bigscience/bloomz-560m')
    BLOOMZ_DEVICE = os.getenv('BLOOMZ_DEVICE', 'auto')  # 'auto', 'cpu', 'cuda'
    BLOOMZ_LOAD_IN_8BIT = os.getenv('BLOOMZ_LOAD_IN_8BIT', 'False').lower() == 'true'
    
    # CORS Configuration
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')
    
    @classmethod
    def get_configured_services(cls):
        """Get list of services that have API keys configured"""
        services = []
        
        if cls.OPENAI_API_KEY:
            services.append('openai')
        if cls.ANTHROPIC_API_KEY:
            services.append('anthropic')
        if cls.DEEPSEEK_API_KEY:
            services.append('deepseek')
        if cls.GEMINI_API_KEY:
            services.append('gemini')
        
        # Bloomz is always available (local model)
        services.append('bloomz')
        
        return services
    
    @classmethod
    def validate_config(cls):
        """Validate configuration and return any warnings"""
        warnings = []
        
        if not cls.OPENAI_API_KEY:
            warnings.append("OpenAI API key not configured")
        if not cls.ANTHROPIC_API_KEY:
            warnings.append("Anthropic API key not configured")
        if not cls.DEEPSEEK_API_KEY:
            warnings.append("DeepSeek API key not configured")
        if not cls.GEMINI_API_KEY:
            warnings.append("Gemini API key not configured")
        
        if cls.DEBUG and cls.SECRET_KEY == 'dev-secret-key-change-in-production':
            warnings.append("Using default secret key in debug mode")
        
        return warnings

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = 'INFO'
    
    # Override with more secure defaults for production
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'https://yourdomain.com').split(',')

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    
    # Use smaller models for testing
    DEFAULT_MODELS = {
        'openai': 'gpt-3.5-turbo',
        'anthropic': 'claude-3-haiku-20240307',
        'deepseek': 'deepseek-chat',
        'gemini': 'gemini-pro',
        'bloomz': 'bigscience/bloomz-560m'
    }

# Configuration mapping
config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config(config_name=None):
    """Get configuration class based on environment"""
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'development')
    
    return config_map.get(config_name, DevelopmentConfig) 