#!/usr/bin/env python
from app import app, init_db
import logging

if __name__ == "__main__":
    # Initialize the database before running
    init_db()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Print all registered routes for debugging
    logging.info("Registered Routes:")
    for rule in app.url_map.iter_rules():
        logging.info(f"{rule} - {rule.methods}")
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5500)
