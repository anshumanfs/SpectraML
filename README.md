![image](https://github.com/user-attachments/assets/0bfaa687-c116-42ae-b60a-d7dabe1743da)

# SpectraML - Comprehensive Data Science Platform

SpectraML is a web-based application for data analysis, visualization, machine learning model training, and image processing. This tool enables data scientists and analysts to work efficiently with various data types through an intuitive interface.

## Features

- **Data Loading**: Support for CSV, XLSX, and JSON files with automatic schema detection
- **Data Visualization**: Interactive Plotly-based charts and dashboards
- **Feature Engineering**: Automated transformation, scaling, encoding, and dimensionality reduction
- **Machine Learning**: Classification, regression, and ensemble models with hyperparameter tuning
- **Deep Learning**: TensorFlow and PyTorch support for neural network training
- **Image Analysis**: Object detection, face recognition, classification, and segmentation
- **Experiment Management**: Track and manage workflows, datasets, and models

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/spectraml.git
   cd spectraml
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   python run.py
   ```

5. Open your web browser and navigate to http://localhost:5000

## Project Structure

- `app.py`: Main Flask application
- `modules/`: Core functionality modules
  - `data_loader.py`: Handles data import and preprocessing
  - `visualization.py`: Creates interactive visualizations
  - `feature_engineering.py`: Implements feature transformations
  - `model_trainer.py`: Provides ML model training capabilities
  - `image_analyzer.py`: Processes and analyzes images
  - `experiment_manager.py`: Manages experiment metadata
- `templates/`: HTML templates for web interface
- `static/`: CSS, JavaScript and static assets
- `storage/`: Storage for experiments, datasets, and models
- `uploads/`: Temporary storage for uploaded files

## Usage

1. Create a new experiment from the dashboard
2. Upload your dataset (CSV, XLSX, JSON)
3. Explore and visualize your data
4. Apply feature engineering operations
5. Train machine learning models
6. Evaluate model performance
7. Deploy or export your models

## License

MIT License - See LICENSE file for details
