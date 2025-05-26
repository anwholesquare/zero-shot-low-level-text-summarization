#!/usr/bin/env python3
"""
Run script for AI Services Flask API
"""

import os
import sys
from app import app
from config import get_config, Config

def main():
    """Main function to run the Flask application"""
    
    # Get configuration
    config_class = get_config()
    config = config_class()
    
    # Print startup information
    print("🚀 AI Services Flask API")
    print("=" * 40)
    print(f"Environment: {os.getenv('FLASK_ENV', 'development')}")
    print(f"Debug Mode: {config.DEBUG}")
    print(f"Host: {config.HOST}")
    print(f"Port: {config.PORT}")
    
    # Validate configuration and show warnings
    warnings = config.validate_config()
    if warnings:
        print("\n⚠️  Configuration Warnings:")
        for warning in warnings:
            print(f"   - {warning}")
    
    # Show configured services
    configured_services = config.get_configured_services()
    print(f"\n✅ Configured Services: {', '.join(configured_services)}")
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    print(f"\n🌐 API will be available at: http://{config.HOST}:{config.PORT}")
    print("📚 API Documentation: Check README.md for endpoint details")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 40)
    
    try:
        # Run the Flask application
        app.run(
            host=config.HOST,
            port=config.PORT,
            debug=config.DEBUG
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 