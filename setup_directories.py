#!/usr/bin/env python3
"""
Setup script to create necessary directories for the XLSUM API
"""

import os

def setup_directories():
    """Create necessary directories"""
    directories = [
        'logs',
        'data',
        'data/samples',
        'data/queues', 
        'data/results'
    ]
    
    print("Setting up directories...")
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created: {directory}")
    
    print("\n🎉 Directory setup completed!")
    print("\nDirectory structure:")
    print("├── data/")
    print("│   ├── samples/     # CSV sample files")
    print("│   ├── queues/      # Queue state files")
    print("│   └── results/     # Processing results")
    print("└── logs/            # Application logs")

if __name__ == "__main__":
    setup_directories() 