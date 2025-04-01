from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    session,
    flash,
    redirect,
    url_for,
)
from werkzeug.utils import secure_filename
from datetime import datetime
import sqlite3
import uuid
import os
import json
import logging
import markdown
import re

from modules.data_loader import DataLoader
from modules.visualization import Visualizer
from modules.feature_engineering import FeatureEngineer
from modules.model_trainer import ModelTrainer
from modules.image_analyzer import ImageAnalyzer
from modules.experiment_manager import ExperimentManager

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config["UPLOAD_FOLDER"] = "uploads/"
app.config["DATABASE"] = "datalab.db"
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB limit

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Ensure upload and storage directories exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs("storage/experiments", exist_ok=True)
os.makedirs("storage/models", exist_ok=True)


# Initialize database
def init_db():
    with sqlite3.connect(app.config["DATABASE"]) as conn:
        conn.execute(
            """
        CREATE TABLE IF NOT EXISTS experiments (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            config TEXT
        )
        """
        )

        conn.execute(
            """
        CREATE TABLE IF NOT EXISTS datasets (
            id TEXT PRIMARY KEY,
            experiment_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            filetype TEXT NOT NULL,
            created_at TEXT NOT NULL,
            metadata TEXT,
            FOREIGN KEY (experiment_id) REFERENCES experiments (id)
        )
        """
        )

        conn.execute(
            """
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
        """
        )


# Initialize components
loader = DataLoader()
visualizer = Visualizer()
feature_engineer = FeatureEngineer()
model_trainer = ModelTrainer()
image_analyzer = ImageAnalyzer()
experiment_manager = ExperimentManager(app.config["DATABASE"])


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/experiments")
def experiments():
    all_experiments = experiment_manager.get_all_experiments()
    print(f"All experiments: {all_experiments}")
    return render_template("experiments.html", experiments=all_experiments)


@app.route("/experiment/new", methods=["GET", "POST"])
def new_experiment():
    if request.method == "POST":
        data = request.form
        exp_id = experiment_manager.create_experiment(
            name=data.get("name"), description=data.get("description")
        )
        return jsonify({"success": True, "experiment_id": exp_id})
    return render_template("new_experiment.html")


@app.route("/experiment/<exp_id>")
def view_experiment(exp_id):
    experiment = experiment_manager.get_experiment(exp_id)
    if not experiment:
        flash("Experiment not found", "error")
        return redirect("/experiments")
    return render_template("experiment.html", experiment=experiment)


@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file part"})

    file = request.files["file"]
    exp_id = request.form.get("experiment_id")

    if not file.filename:
        return jsonify({"success": False, "error": "No selected file"})

    # Secure and save the file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # Register dataset with experiment
    file_type = filename.split(".")[-1].lower()
    dataset_id = experiment_manager.add_dataset(
        experiment_id=exp_id, filename=filename, filetype=file_type
    )

    # Load initial data info
    try:
        data_info = loader.get_data_info(file_path, file_type)
        return jsonify(
            {"success": True, "dataset_id": dataset_id, "file_info": data_info}
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/visualize", methods=["POST"])
def visualize_data():
    data = request.json
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], data.get("filename"))
    viz_type = data.get("viz_type")
    params = data.get("params", {})

    try:
        result = visualizer.create_visualization(file_path, viz_type, params)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/feature-engineering", methods=["POST"])
def apply_feature_engineering():
    """Apply feature engineering operations and save the result as a new dataset"""
    data = request.json
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], data.get("filename"))
    operations = data.get("operations", [])
    experiment_id = data.get("experiment_id")

    try:
        # Apply operations and save result
        result = feature_engineer.apply_operations(file_path, operations)

        # Register the new processed dataset with the experiment
        processed_filename = os.path.basename(result["output_file"])
        file_type = processed_filename.split(".")[-1].lower()

        # Add metadata about processing history
        metadata = {
            "source_file": data.get("filename"),
            "processing_date": datetime.now().isoformat(),
            "operations": operations,
            "processing_stats": {
                "original_rows": result["data_info"]["num_rows"],
                "original_columns": result["data_info"]["num_columns"],
            },
        }

        # Add processed dataset to experiment
        dataset_id = experiment_manager.add_dataset(
            experiment_id=experiment_id,
            filename=processed_filename,
            filetype=file_type,
            metadata=json.dumps(metadata),
        )

        return jsonify(
            {
                "success": True,
                "dataset_id": dataset_id,
                "filename": processed_filename,
                "data_info": result["data_info"],
            }
        )
    except Exception as e:
        logging.error(f"Error applying feature engineering: {str(e)}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/train-model", methods=["POST"])
def train_model():
    data = request.json
    exp_id = data.get("experiment_id")
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], data.get("filename"))
    model_type = data.get("model_type")
    config = data.get("config", {})

    try:
        result = model_trainer.train(file_path, model_type, config)

        # Save model info to database
        model_id = experiment_manager.add_model(
            experiment_id=exp_id,
            name=data.get("model_name", f"Model-{uuid.uuid4().hex[:8]}"),
            model_type=model_type,
            metrics=json.dumps(result.get("metrics", {})),
            config=json.dumps(config),
        )

        result["model_id"] = model_id
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error training model: {str(e)}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/image-analysis", methods=["POST"])
def analyze_image():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided"})

    image_file = request.files["image"]
    analysis_type = request.form.get("analysis_type")

    try:
        result = image_analyzer.analyze(image_file, analysis_type)
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error analyzing image: {str(e)}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/experiment/<exp_id>", methods=["DELETE"])
def delete_experiment(exp_id):
    """Delete an experiment"""
    try:
        success = experiment_manager.delete_experiment(exp_id)
        if success:
            return jsonify(
                {"success": True, "message": "Experiment deleted successfully"}
            )
        else:
            return jsonify({"success": False, "error": "Failed to delete experiment"})
    except Exception as e:
        logging.error(f"Error deleting experiment: {str(e)}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/experiment/<exp_id>/dataset/<dataset_id>", methods=["DELETE"])
def delete_dataset(exp_id, dataset_id):
    """Delete a dataset from an experiment"""
    # Add logging to debug the issue
    logging.info(f"Deleting dataset {dataset_id} from experiment {exp_id}")
    try:
        success = experiment_manager.delete_dataset(exp_id, dataset_id)
        if success:
            return jsonify({"success": True, "message": "Dataset deleted successfully"})
        else:
            return jsonify({"success": False, "error": "Failed to delete dataset"})
    except Exception as e:
        logging.error(f"Error deleting dataset: {str(e)}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/experiment/<exp_id>/model/<model_id>", methods=["DELETE"])
def delete_model(exp_id, model_id):
    """Delete a model from an experiment"""
    try:
        # Add logging for debugging
        logging.info(f"Deleting model {model_id} from experiment {exp_id}")

        # Call the experiment manager to delete the model
        success = experiment_manager.delete_model(exp_id, model_id)

        if success:
            return jsonify({"success": True, "message": "Model deleted successfully"})
        else:
            return jsonify({"success": False, "error": "Failed to delete model"})
    except Exception as e:
        logging.error(f"Error deleting model: {str(e)}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/dataset/<dataset_id>/visualize")
def visualize_dataset(dataset_id):
    """Visualization page for a specific dataset"""
    # Get dataset information
    dataset = experiment_manager.get_dataset(dataset_id)
    if not dataset:
        flash("Dataset not found", "error")
        return redirect("/experiments")

    # Get experiment for this dataset
    experiment = experiment_manager.get_experiment(dataset["experiment_id"])

    # Get file path
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], dataset["filename"])

    # Load initial data info for display
    data_info = None
    try:
        data_info = loader.get_data_info(file_path, dataset["filetype"])
    except Exception as e:
        logging.error(f"Error loading data info: {str(e)}")
        flash(f"Error loading data: {str(e)}", "error")

    return render_template(
        "dataset_visualize.html",
        dataset=dataset,
        experiment=experiment,
        data_info=data_info,
    )


@app.route("/experiment/<exp_id>/visualize")
def visualize_experiment(exp_id):
    """Visualization creation page for an experiment"""
    experiment = experiment_manager.get_experiment(exp_id)
    if not experiment:
        flash("Experiment not found", "error")
        return redirect("/experiments")

    # Get datasets for this experiment
    datasets = experiment.get("datasets", [])
    if not datasets:
        flash("No datasets available. Upload data first.", "warning")
        return redirect(f"/experiment/{exp_id}")

    return render_template(
        "experiment_visualize.html", experiment=experiment, datasets=datasets
    )


@app.route("/experiment/<exp_id>/feature-engineering")
def feature_engineering_page(exp_id):
    """Feature engineering page for an experiment"""
    experiment = experiment_manager.get_experiment(exp_id)
    if not experiment:
        flash("Experiment not found", "error")
        return redirect("/experiments")

    # Get datasets for this experiment
    datasets = experiment.get("datasets", [])

    # Get available feature engineering operations
    available_operations = feature_engineer.supported_operations

    return render_template(
        "feature_engineering.html",
        experiment=experiment,
        datasets=datasets,
        available_operations=available_operations,
    )


@app.route("/experiment/<exp_id>/train-model")
def train_model_page(exp_id):
    """Model training page for an experiment"""
    experiment = experiment_manager.get_experiment(exp_id)
    if not experiment:
        flash("Experiment not found", "error")
        return redirect("/experiments")

    # Get datasets for this experiment
    datasets = experiment.get("datasets", [])

    # For each dataset, get column information if possible
    for dataset in datasets:
        try:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], dataset["filename"])
            data_info = loader.get_data_info(file_path, dataset["filetype"])
            dataset["columns"] = data_info.get("columns", [])
        except Exception as e:
            logging.error(
                f"Error loading column info for dataset {dataset['id']}: {str(e)}"
            )
            dataset["columns"] = []

    return render_template("train_model.html", experiment=experiment, datasets=datasets)


@app.route("/api/feature-engineering/preview", methods=["POST"])
def preview_feature_engineering():
    """Preview the results of feature engineering operations"""
    data = request.json
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], data.get("filename"))
    operations = data.get("operations", [])

    try:
        # Call feature engineer to preview operations
        result = feature_engineer.preview_operations(file_path, operations, max_rows=10)
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error previewing feature engineering: {str(e)}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/dataset/preview/<dataset_id>", methods=["GET"])
def preview_dataset(dataset_id):
    """Get a preview of a dataset"""
    try:
        # Get dataset information
        dataset = experiment_manager.get_dataset(dataset_id)
        if not dataset:
            return jsonify({"success": False, "error": "Dataset not found"})

        # Get file path
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], dataset["filename"])

        # Use DataLoader to get preview
        file_type = dataset["filetype"]
        preview_data = loader.get_preview_data(file_path, file_type, max_rows=10)

        return jsonify({"success": True, "dataset": dataset, "preview": preview_data})
    except Exception as e:
        logging.error(f"Error getting dataset preview: {str(e)}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/dataset/columns/<dataset_id>", methods=["GET"])
def get_dataset_columns(dataset_id):
    """Get columns for a dataset"""
    try:
        # Get dataset information
        dataset = experiment_manager.get_dataset(dataset_id)
        if not dataset:
            return jsonify({"success": False, "error": "Dataset not found"})

        # Get file path
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], dataset["filename"])

        # Use DataLoader to get data info
        file_type = dataset["filetype"]
        data_info = loader.get_data_info(file_path, file_type)

        return jsonify({"success": True, "columns": data_info["columns"]})
    except Exception as e:
        logging.error(f"Error getting dataset columns: {str(e)}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/model/<model_id>")
def view_model(model_id):
    """View model details page"""
    try:
        # Get model information
        model = experiment_manager.get_model(model_id)
        if not model:
            flash("Model not found", "error")
            return redirect("/experiments")

        # Get experiment information
        experiment = experiment_manager.get_experiment(model["experiment_id"])

        return render_template("model_details.html", model=model, experiment=experiment)
    except Exception as e:
        logging.error(f"Error viewing model: {str(e)}")
        flash(f"Error loading model details: {str(e)}", "error")
        return redirect("/experiments")


@app.route("/api/visualization/options/<viz_type>", methods=["GET"])
def get_visualization_options(viz_type):
    """Get options for a specific visualization type"""
    try:
        options = visualizer.get_visualization_options(viz_type)
        return jsonify({"success": True, "options": options})
    except Exception as e:
        logging.error(f"Error getting visualization options: {str(e)}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/visualization/save", methods=["POST"])
def save_visualization():
    """Save a visualization for an experiment"""
    data = request.json
    experiment_id = data.get("experiment_id")
    dataset_id = data.get("dataset_id")
    viz_type = data.get("viz_type")
    params = data.get("params", {})
    title = data.get("title", f"{viz_type.replace('_', ' ').title()} Visualization")

    try:
        # Create a visualization ID
        viz_id = uuid.uuid4().hex

        # Save visualization data to database or filesystem
        # Implementation depends on how you want to persist visualizations

        return jsonify({"success": True, "visualization_id": viz_id})
    except Exception as e:
        logging.error(f"Error saving visualization: {str(e)}")
        return jsonify({"success": False, "error": str(e)})


# Add the following utility function to convert markdown links to proper URL paths
def fix_markdown_links(content, current_file=None):
    """
    Converts relative markdown links to proper Flask route links

    Args:
        content (str): The markdown content
        current_file (str): The current file name (without .md extension)

    Returns:
        str: Updated content with fixed links
    """
    import re

    # Replace links like [text](file.md) with [text](/ml-guide/file)
    # But don't modify external links with http:// or https://
    pattern = r"\[([^\]]+)\]\((?!http)(.*?)\.md\)"

    def replace_link(match):
        text = match.group(1)
        link = match.group(2)

        # Skip if it's already an absolute URL
        if link.startswith("/"):
            return f"[{text}]({link})"

        # Handle index specially
        if link == "index":
            return f"[{text}](/ml-guide)"

        # Make the link absolute
        return f"[{text}](/ml-guide/{link})"

    # Apply the regex replacement
    updated_content = re.sub(pattern, replace_link, content)

    return updated_content


@app.route("/ml-guide")
@app.route("/ml-guide/")
def ml_guide_index():
    """Render the machine learning guide index"""
    try:
        # Path to the index markdown file
        docs_dir = os.path.join(app.root_path, "docs")
        logger.info(f"Looking for markdown files in: {docs_dir}")

        # Make sure docs directory exists
        if not os.path.exists(docs_dir):
            logger.error(f"Documentation directory not found: {docs_dir}")
            flash("Documentation directory not found.", "error")
            return redirect("/")

        # List available markdown files for debugging
        available_files = [f for f in os.listdir(docs_dir) if f.endswith(".md")]
        logger.info(f"Available markdown files: {available_files}")

        md_file_path = os.path.join(docs_dir, "index.md")

        # If the index doesn't exist, try the legacy single file or a fallback
        if not os.path.exists(md_file_path):
            logger.warning(f"Index file not found: {md_file_path}")

            # Try ml_guide.md as fallback
            ml_guide_path = os.path.join(docs_dir, "ml_guide.md")
            if os.path.exists(ml_guide_path):
                logger.info(f"Using ml_guide.md as fallback")
                md_file_path = ml_guide_path
            else:
                # If no markdown files exist, create a simple one
                if not available_files:
                    logger.warning("No markdown files found. Creating a placeholder.")
                    with open(md_file_path, "w") as f:
                        f.write(
                            "# Machine Learning Guide\n\nWelcome to the SpectraML Machine Learning Guide. This documentation is currently under development."
                        )
                else:
                    # Use the first available markdown file
                    md_file_path = os.path.join(docs_dir, available_files[0])
                    logger.info(
                        f"Using first available file as fallback: {available_files[0]}"
                    )

        # Read the markdown file
        logger.info(f"Reading markdown file: {md_file_path}")
        with open(md_file_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # Fix the links in the content
        md_content = fix_markdown_links(md_content, "index")

        # Convert markdown to HTML
        logger.info("Converting markdown to HTML")
        html_content = markdown.markdown(
            md_content, extensions=["tables", "fenced_code", "codehilite", "toc"]
        )

        # Get guide files for navigation
        guide_files = []
        for file in available_files:
            if file != "index.md" and file != "ml_guide.md":
                name = file[:-3]  # Remove .md extension
                title = name.replace("_", " ").title()
                guide_files.append({"name": name, "title": title, "file": file})

        # Sort guide files in a logical order
        file_order = [
            "introduction",
            "feature_engineering",
            "model_selection",
            "model_evaluation",
            "deep_learning",
            "spectral_data",
            "advanced_topics",
            "best_practices",
            "glossary",
            "code_examples",
            "references",
        ]
        guide_files.sort(
            key=lambda x: (
                file_order.index(x["name"]) if x["name"] in file_order else 999
            )
        )

        logger.info(f"Found {len(guide_files)} guide files for navigation")

        # Render template with HTML content
        current_file = "index"
        if md_file_path.endswith("ml_guide.md"):
            current_file = "ml_guide"

        return render_template(
            "markdown_view.html",
            title="Machine Learning Guide",
            content=html_content,
            guide_files=guide_files,
            current_file=current_file,
            debug_info={
                "app_root": app.root_path,
                "docs_dir": docs_dir,
                "md_file": md_file_path,
            },
        )
    except Exception as e:
        logger.error(f"Error rendering ML guide index: {str(e)}", exc_info=True)
        flash(f"Error loading guide: {str(e)}", "error")
        return redirect("/ml-guide")


@app.route("/ml-guide/<guide_file>")
def ml_guide_page(guide_file):
    """Render a specific machine learning guide page"""
    try:
        # Make sure the filename is safe
        guide_file = guide_file.replace("..", "").replace("/", "")

        # Add .md extension if not already present
        if not guide_file.endswith(".md"):
            guide_file = f"{guide_file}.md"

        # Path to the markdown file
        docs_dir = os.path.join(app.root_path, "docs")
        md_file_path = os.path.join(docs_dir, guide_file)

        logger.info(f"Looking for guide file: {md_file_path}")

        # Check if file exists
        if not os.path.exists(md_file_path):
            logger.warning(f"Guide page not found: {md_file_path}")

            # List available markdown files for debugging
            available_files = [f for f in os.listdir(docs_dir) if f.endswith(".md")]
            logger.info(f"Available markdown files: {available_files}")

            flash(f"Guide page not found: {guide_file}", "error")
            return redirect("/ml-guide")

        # Read the markdown file
        with open(md_file_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # Fix the links in the content
        md_content = fix_markdown_links(
            md_content, guide_file[:-3] if guide_file.endswith(".md") else guide_file
        )

        # Convert markdown to HTML
        html_content = markdown.markdown(
            md_content, extensions=["tables", "fenced_code", "codehilite", "toc"]
        )

        # Get all guide files for navigation
        guide_files = []
        for file in os.listdir(docs_dir):
            if file.endswith(".md") and file != "index.md" and file != "ml_guide.md":
                name = file[:-3]  # Remove .md extension
                title = name.replace("_", " ").title()
                guide_files.append({"name": name, "title": title, "file": file})

        # Sort guide files in a logical order
        file_order = [
            "introduction",
            "feature_engineering",
            "model_selection",
            "model_evaluation",
            "deep_learning",
            "spectral_data",
            "advanced_topics",
            "best_practices",
            "glossary",
            "code_examples",
            "references",
        ]
        guide_files.sort(
            key=lambda x: (
                file_order.index(x["name"]) if x["name"] in file_order else 999
            )
        )

        # Get the title from heading if possible
        title = guide_file[:-3].replace("_", " ").title()
        heading_match = re.search(r"^# (.+)$", md_content, re.MULTILINE)
        if heading_match:
            title = heading_match.group(1)

        # Render template with HTML content
        return render_template(
            "markdown_view.html",
            title=title,
            content=html_content,
            guide_files=guide_files,
            current_file=guide_file[:-3],  # Remove .md for comparison
            debug_info={
                "app_root": app.root_path,
                "docs_dir": docs_dir,
                "md_file": md_file_path,
            },
        )
    except Exception as e:
        logger.error(f"Error rendering ML guide page: {str(e)}", exc_info=True)
        flash(f"Error loading guide page: {str(e)}", "error")
        return redirect("/ml-guide")


# Create a custom Jinja2 filter for datetime formatting
@app.template_filter("datetime")
def format_datetime(value, format="%B %d, %Y at %H:%M"):
    """Format a datetime string to a readable format"""
    if value is None:
        return ""

    if isinstance(value, str):
        try:
            # Try to parse ISO format
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.strftime(format)
        except (ValueError, AttributeError):
            # If parsing fails, return the original string
            return value

    # If value is already a datetime object
    if isinstance(value, datetime):
        return value.strftime(format)

    # Default fallback
    return str(value)


# Add a custom Jinja2 filter for JSON parsing
@app.template_filter("fromjson")
def parse_json(value):
    """Parse a JSON string into a Python object"""
    if value is None or value == "":
        return {}

    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            # If JSON parsing fails, return empty dict
            return {}

    # If value is already a dict or other object, return as is
    return value


# Add a custom Jinja2 filter for JSON to string conversion
@app.template_filter("tojson")
def to_json_filter(value, indent=None):
    """Convert a Python object to a JSON string with optional indentation"""
    if value is None:
        return "{}"
    return json.dumps(value, indent=indent)


if __name__ == "__main__":
    init_db()

    # Print all registered routes for debugging
    logging.basicConfig(level=logging.INFO)
    logging.info("Registered Routes:")
    for rule in app.url_map.iter_rules():
        logging.info(f"{rule} - {rule.methods}")

    app.run(debug=True)
