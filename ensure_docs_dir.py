"""
Utility script to ensure the docs directory exists and contains at least one markdown file.
Run this script if you're having issues with the documentation not displaying.
"""

import os
import sys
import shutil
from pathlib import Path

def ensure_docs_directory():
    """Ensure docs directory exists with at least one markdown file."""
    # Get the root project directory
    project_dir = Path(__file__).parent
    docs_dir = project_dir / 'docs'
    
    # Create docs directory if it doesn't exist
    if not docs_dir.exists():
        print(f"Creating docs directory at {docs_dir}")
        docs_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if the directory is empty or missing required files
    md_files = list(docs_dir.glob('*.md'))
    if not md_files:
        print("No markdown files found in docs directory. Creating index.md")
        
        # Create a basic index.md file
        index_path = docs_dir / 'index.md'
        with open(index_path, 'w') as f:
            f.write("""# SpectraML Documentation

Welcome to the SpectraML documentation. This guide provides information on machine learning concepts and techniques for spectral data analysis.

## Quick Start

This is a placeholder index page. The complete documentation will be available soon.
""")
        print(f"Created {index_path}")
    else:
        print(f"Found {len(md_files)} markdown files in docs directory:")
        for file in md_files:
            print(f"  - {file.name}")
    
    # Check that index.md exists
    index_path = docs_dir / 'index.md'
    if not index_path.exists():
        print("index.md not found. Creating it from another markdown file if available")
        if md_files:
            # Copy the first available markdown file to index.md
            shutil.copy(md_files[0], index_path)
            print(f"Created index.md by copying {md_files[0].name}")
        else:
            print("Unable to create index.md as no markdown files are available")
    
    print("\nDocs directory check completed.")
    return True

if __name__ == "__main__":
    ensure_docs_directory()
    print("\nTo test the documentation, run the Flask app and navigate to /ml-guide")
