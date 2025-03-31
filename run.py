#!/usr/bin/env python
import os
import sys
import logging
from app import app, init_db

# Initialize the database before running
init_db()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('spectra-ml.log')
    ]
)

# Check if required directories exist
required_dirs = [
    'uploads',
    'storage',
    'storage/experiments',
    'storage/models',
    'docs'
]

for directory in required_dirs:
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Created directory: {directory}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5500))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logging.info(f"Starting SpectraML on port {port} (debug={debug})")
    app.run(host='0.0.0.0', port=port, debug=debug)
