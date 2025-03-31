import sqlite3
import json
import uuid
from datetime import datetime
import os

class ExperimentManager:
    """
    Manages experiments, datasets, and models in the application.
    """
    
    def __init__(self, db_path):
        """
        Initialize the experiment manager
        
        Parameters:
        -----------
        db_path : str
            Path to the SQLite database
        """
        self.db_path = db_path
    
    def get_db_connection(self):
        """Get a connection to the database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        return conn
    
    def create_experiment(self, name, description=""):
        """
        Create a new experiment
        
        Parameters:
        -----------
        name : str
            Name of the experiment
        description : str, optional
            Description of the experiment
            
        Returns:
        --------
        str
            ID of the created experiment
        """
        exp_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        conn = self.get_db_connection()
        conn.execute(
            "INSERT INTO experiments (id, name, description, created_at, updated_at, config) VALUES (?, ?, ?, ?, ?, ?)",
            (exp_id, name, description, now, now, '{}')
        )
        conn.commit()
        conn.close()
        
        # Create experiment directory
        os.makedirs(f"storage/experiments/{exp_id}", exist_ok=True)
        
        return exp_id
    
    def get_experiment(self, exp_id):
        """
        Get an experiment by ID
        
        Parameters:
        -----------
        exp_id : str
            ID of the experiment
            
        Returns:
        --------
        dict
            Experiment data including datasets and models
        """
        conn = self.get_db_connection()
        
        # Get experiment
        exp = conn.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,)).fetchone()
        
        if not exp:
            conn.close()
            return None
        
        # Convert to dict
        experiment = dict(exp)
        experiment['config'] = json.loads(experiment['config'])
        
        # Get datasets for this experiment
        datasets = conn.execute("SELECT * FROM datasets WHERE experiment_id = ?", (exp_id,)).fetchall()
        experiment['datasets'] = [dict(d) for d in datasets]
        
        # Parse metadata for each dataset
        for dataset in experiment['datasets']:
            if dataset['metadata']:
                dataset['metadata'] = json.loads(dataset['metadata'])
        
        # Get models for this experiment
        models = conn.execute("SELECT * FROM models WHERE experiment_id = ?", (exp_id,)).fetchall()
        experiment['models'] = [dict(m) for m in models]
        
        # Parse metrics and config for each model
        for model in experiment['models']:
            if model['metrics']:
                model['metrics'] = json.loads(model['metrics'])
            if model['config']:
                model['config'] = json.loads(model['config'])
        
        conn.close()
        return experiment
    
    def get_all_experiments(self):
        """
        Get all experiments
        
        Returns:
        --------
        list
            List of all experiments
        """
        conn = self.get_db_connection()
        
        # Get all experiments
        exps = conn.execute("SELECT * FROM experiments ORDER BY created_at DESC").fetchall()
        
        # Convert to list of dicts
        experiments = []
        for exp in exps:
            exp_dict = dict(exp)
            exp_dict['config'] = json.loads(exp_dict['config'])
            
            # Get dataset count
            dataset_count = conn.execute(
                "SELECT COUNT(*) FROM datasets WHERE experiment_id = ?", (exp_dict['id'],)
            ).fetchone()[0]
            exp_dict['dataset_count'] = dataset_count
            
            # Get model count
            model_count = conn.execute(
                "SELECT COUNT(*) FROM models WHERE experiment_id = ?", (exp_dict['id'],)
            ).fetchone()[0]
            exp_dict['model_count'] = model_count
            
            experiments.append(exp_dict)
        
        conn.close()
        return experiments
    
    def update_experiment(self, exp_id, updates):
        """
        Update an experiment
        
        Parameters:
        -----------
        exp_id : str
            ID of the experiment to update
        updates : dict
            Updates to apply to the experiment
            
        Returns:
        --------
        bool
            True if update was successful, False otherwise
        """
        conn = self.get_db_connection()
        
        # Get current experiment
        exp = conn.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,)).fetchone()
        
        if not exp:
            conn.close()
            return False
        
        # Update timestamp
        updates['updated_at'] = datetime.now().isoformat()
        
        # Build SET clause and parameters for SQL query
        set_clause = ", ".join([f"{key} = ?" for key in updates.keys()])
        params = list(updates.values()) + [exp_id]
        
        # Execute update
        conn.execute(
            f"UPDATE experiments SET {set_clause} WHERE id = ?",
            params
        )
        
        conn.commit()
        conn.close()
        
        return True
    
    def delete_experiment(self, exp_id):
        """
        Delete an experiment and all associated datasets and models
        
        Parameters:
        -----------
        exp_id : str
            ID of the experiment to delete
            
        Returns:
        --------
        bool
            True if deletion was successful, False otherwise
        """
        conn = self.get_db_connection()
        
        # Check if experiment exists
        exp = conn.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,)).fetchone()
        
        if not exp:
            conn.close()
            return False
        
        try:
            # Delete associated datasets
            conn.execute("DELETE FROM datasets WHERE experiment_id = ?", (exp_id,))
            
            # Delete associated models
            conn.execute("DELETE FROM models WHERE experiment_id = ?", (exp_id,))
            
            # Delete experiment
            conn.execute("DELETE FROM experiments WHERE id = ?", (exp_id,))
            
            conn.commit()
            
            # Delete experiment directory if it exists
            exp_dir = f"storage/experiments/{exp_id}"
            if os.path.exists(exp_dir):
                import shutil
                shutil.rmtree(exp_dir)
                
            return True
        except Exception as e:
            conn.rollback()
            print(f"Error deleting experiment: {e}")
            return False
        finally:
            conn.close()
    
    def add_dataset(self, experiment_id, filename, filetype, metadata=None):
        """
        Add a dataset to an experiment
        
        Parameters:
        -----------
        experiment_id : str
            ID of the experiment
        filename : str
            Name of the dataset file
        filetype : str
            Type of the dataset file
        metadata : dict, optional
            Metadata for the dataset
            
        Returns:
        --------
        str
            ID of the created dataset
        """
        dataset_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        # Convert metadata to JSON string
        metadata_json = json.dumps(metadata) if metadata else None
        
        conn = self.get_db_connection()
        
        # Check if experiment exists
        exp = conn.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
        
        if not exp:
            conn.close()
            raise ValueError(f"Experiment with ID {experiment_id} not found")
        
        conn.execute(
            "INSERT INTO datasets (id, experiment_id, filename, filetype, created_at, metadata) VALUES (?, ?, ?, ?, ?, ?)",
            (dataset_id, experiment_id, filename, filetype, now, metadata_json)
        )
        
        # Update experiment's updated_at timestamp
        conn.execute(
            "UPDATE experiments SET updated_at = ? WHERE id = ?",
            (now, experiment_id)
        )
        
        conn.commit()
        conn.close()
        
        return dataset_id
    
    def add_model(self, experiment_id, name, model_type, metrics=None, config=None):
        """
        Add a model to an experiment
        
        Parameters:
        -----------
        experiment_id : str
            ID of the experiment
        name : str
            Name of the model
        model_type : str
            Type of the model
        metrics : dict or str, optional
            Metrics for the model (will be converted to JSON if dict)
        config : dict or str, optional
            Configuration for the model (will be converted to JSON if dict)
            
        Returns:
        --------
        str
            ID of the created model
        """
        model_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        # Convert metrics and config to JSON strings if they are dicts
        if metrics and isinstance(metrics, dict):
            metrics = json.dumps(metrics)
        if config and isinstance(config, dict):
            config = json.dumps(config)
        
        conn = self.get_db_connection()
        
        # Check if experiment exists
        exp = conn.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
        
        if not exp:
            conn.close()
            raise ValueError(f"Experiment with ID {experiment_id} not found")
        
        conn.execute(
            "INSERT INTO models (id, experiment_id, name, model_type, created_at, metrics, config) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (model_id, experiment_id, name, model_type, now, metrics, config)
        )
        
        # Update experiment's updated_at timestamp
        conn.execute(
            "UPDATE experiments SET updated_at = ? WHERE id = ?",
            (now, experiment_id)
        )
        
        conn.commit()
        conn.close()
        
        return model_id
    
    def get_dataset(self, dataset_id):
        """
        Get a dataset by ID
        
        Parameters:
        -----------
        dataset_id : str
            ID of the dataset
            
        Returns:
        --------
        dict
            Dataset data
        """
        conn = self.get_db_connection()
        
        # Get dataset
        dataset = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
        
        if not dataset:
            conn.close()
            return None
        
        # Convert to dict
        dataset_dict = dict(dataset)
        
        # Parse metadata if it exists
        if dataset_dict['metadata']:
            dataset_dict['metadata'] = json.loads(dataset_dict['metadata'])
        
        conn.close()
        return dataset_dict
    
    def get_model(self, model_id):
        """
        Get a model by ID
        
        Parameters:
        -----------
        model_id : str
            ID of the model
            
        Returns:
        --------
        dict or None
            Model information or None if not found
        """
        try:
            conn = self.get_db_connection()
            model = conn.execute(
                "SELECT * FROM models WHERE id = ?", 
                (model_id,)
            ).fetchone()
            conn.close()
            
            if model:
                # Convert SQLite row to dict
                model_dict = dict(model)
                return model_dict
            return None
        except Exception as e:
            print(f"Error getting model: {str(e)}")
            return None
        
    def delete_dataset(self, experiment_id, dataset_id):
        """
        Delete a dataset from an experiment
        
        Parameters:
        -----------
        experiment_id : str
            ID of the experiment
        dataset_id : str
            ID of the dataset to delete
            
        Returns:
        --------
        bool
            True if deletion was successful, False otherwise
        """
        conn = self.get_db_connection()
        
        # Check if experiment exists
        exp = conn.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
        
        if not exp:
            conn.close()
            return False
        
        # Check if dataset exists and belongs to the experiment
        dataset = conn.execute(
            "SELECT * FROM datasets WHERE id = ? AND experiment_id = ?", 
            (dataset_id, experiment_id)
        ).fetchone()
        
        if not dataset:
            conn.close()
            return False
        
        try:
            # Get dataset details for file deletion
            dataset_dict = dict(dataset)
            filename = dataset_dict['filename']
            
            # Delete dataset from database
            conn.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
            
            # Update experiment's updated_at timestamp
            now = datetime.now().isoformat()
            conn.execute(
                "UPDATE experiments SET updated_at = ? WHERE id = ?",
                (now, experiment_id)
            )
            
            conn.commit()
            
            # Delete the actual dataset file if it exists
            file_path = f"uploads/{filename}"
            if os.path.exists(file_path):
                os.remove(file_path)
                
            return True
        except Exception as e:
            conn.rollback()
            print(f"Error deleting dataset: {e}")
            return False
        finally:
            conn.close()
    
    def delete_model(self, experiment_id, model_id):
        """
        Delete a model from an experiment
        
        Parameters:
        -----------
        experiment_id : str
            ID of the experiment
        model_id : str
            ID of the model to delete
            
        Returns:
        --------
        bool
            True if deletion was successful, False otherwise
        """
        try:
            # First, get the model information to find the model file
            conn = self.get_db_connection()
            model = conn.execute(
                "SELECT * FROM models WHERE id = ? AND experiment_id = ?",
                (model_id, experiment_id)
            ).fetchone()
            
            if not model:
                conn.close()
                raise ValueError(f"Model {model_id} not found in experiment {experiment_id}")
            
            # Get model file path from the stored configuration if available
            model_path = None
            if model['config']:
                try:
                    config = json.loads(model['config'])
                    model_path = config.get('model_path')
                except json.JSONDecodeError:
                    # If config is not valid JSON, continue without model file deletion
                    pass
            
            # Delete the model record from the database
            conn.execute(
                "DELETE FROM models WHERE id = ? AND experiment_id = ?",
                (model_id, experiment_id)
            )
            conn.commit()
            conn.close()
            
            # Delete the model file if path was found
            if model_path and os.path.exists(model_path):
                try:
                    os.remove(model_path)
                    print(f"Deleted model file: {model_path}")
                except Exception as e:
                    print(f"Warning: Failed to delete model file {model_path}: {str(e)}")
                    # Continue even if file deletion fails
            
            return True
        except Exception as e:
            print(f"Error deleting model: {str(e)}")
            return False
