from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
import os
import json
import uuid
import sqlite3
from datetime import datetime

from modules.data_loader import DataLoader
from modules.visualization import Visualizer
from modules.feature_engineering import FeatureEngineer
from modules.model_trainer import ModelTrainer
from modules.image_analyzer import ImageAnalyzer
from modules.experiment_manager import ExperimentManager

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['DATABASE'] = 'datalab.db'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit

# Ensure upload and storage directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('storage/experiments', exist_ok=True)
os.makedirs('storage/models', exist_ok=True)

# Initialize database
def init_db():
    with sqlite3.connect(app.config['DATABASE']) as conn:
        conn.execute('''
        CREATE TABLE IF NOT EXISTS experiments (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            config TEXT
        )
        ''')
        
        conn.execute('''
        CREATE TABLE IF NOT EXISTS datasets (
            id TEXT PRIMARY KEY,
            experiment_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            filetype TEXT NOT NULL,
            created_at TEXT NOT NULL,
            metadata TEXT,
            FOREIGN KEY (experiment_id) REFERENCES experiments (id)
        )
        ''')
        
        conn.execute('''
        CREATE TABLE IF NOT EXISTS models (
            id TEXT PRIMARY KEY,
            experiment_id TEXT NOT NULL,
            name TEXT NOT NULL,
            model_type TEXT NOT NULL,
            created_at TEXT NOT NULL,
            metrics TEXT,
            config TEXT,
            FOREIGN KEY (experiment_id) REFERENCES experiments (id)
        )
        ''')

# Initialize components
loader = DataLoader()
visualizer = Visualizer()
feature_engineer = FeatureEngineer()
model_trainer = ModelTrainer()
image_analyzer = ImageAnalyzer()
experiment_manager = ExperimentManager(app.config['DATABASE'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/experiments')
def experiments():
    all_experiments = experiment_manager.get_all_experiments()
    return render_template('experiments.html', experiments=all_experiments)

@app.route('/experiment/new', methods=['GET', 'POST'])
def new_experiment():
    if request.method == 'POST':
        data = request.form
        exp_id = experiment_manager.create_experiment(
            name=data.get('name'),
            description=data.get('description')
        )
        return jsonify({'success': True, 'experiment_id': exp_id})
    return render_template('new_experiment.html')

@app.route('/experiment/<exp_id>')
def view_experiment(exp_id):
    experiment = experiment_manager.get_experiment(exp_id)
    if not experiment:
        return "Experiment not found", 404
    return render_template('experiment.html', experiment=experiment)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    exp_id = request.form.get('experiment_id')
    
    if not file.filename:
        return jsonify({'error': 'No selected file'}), 400
    
    # Secure and save the file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    # Register dataset with experiment
    file_type = filename.split('.')[-1].lower()
    dataset_id = experiment_manager.add_dataset(
        experiment_id=exp_id,
        filename=filename,
        filetype=file_type
    )
    
    # Load initial data info
    try:
        data_info = loader.get_data_info(file_path, file_type)
        return jsonify({
            'success': True,
            'dataset_id': dataset_id,
            'file_info': data_info
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/visualize', methods=['POST'])
def visualize_data():
    data = request.json
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], data.get('filename'))
    viz_type = data.get('viz_type')
    params = data.get('params', {})
    
    try:
        result = visualizer.create_visualization(file_path, viz_type, params)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/feature-engineering', methods=['POST'])
def engineer_features():
    data = request.json
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], data.get('filename'))
    operations = data.get('operations', [])
    
    try:
        result = feature_engineer.apply_operations(file_path, operations)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train-model', methods=['POST'])
def train_model():
    data = request.json
    exp_id = data.get('experiment_id')
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], data.get('filename'))
    model_type = data.get('model_type')
    config = data.get('config', {})
    
    try:
        result = model_trainer.train(file_path, model_type, config)
        
        # Save model info to database
        model_id = experiment_manager.add_model(
            experiment_id=exp_id,
            name=data.get('model_name', f"Model-{uuid.uuid4().hex[:8]}"),
            model_type=model_type,
            metrics=json.dumps(result.get('metrics', {})),
            config=json.dumps(config)
        )
        
        result['model_id'] = model_id
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/image-analysis', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file'}), 400
    
    image_file = request.files['image']
    analysis_type = request.form.get('analysis_type')
    
    try:
        result = image_analyzer.analyze(image_file, analysis_type)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/experiment/<exp_id>', methods=['DELETE'])
def delete_experiment(exp_id):
    """Delete an experiment"""
    try:
        success = experiment_manager.delete_experiment(exp_id)
        if success:
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Failed to delete experiment'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
