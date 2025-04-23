from roboflow import Roboflow
import argparse
import shutil
from pathlib import Path
from config import get_env_var, select_project

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Download dataset from Roboflow')
    parser.add_argument('--project', type=str, help='Roboflow project name')
    parser.add_argument('--version', type=int, help='Roboflow dataset version')
    parser.add_argument('--api-key', type=str, help='Roboflow API key')
    parser.add_argument('--dataset-format', type=str, default='yolov11', help='Dataset format')
    parser.add_argument('--dataset-path', type=str, help='Dataset path')
    args = parser.parse_args()

    # Get API key from command line arguments or environment variables
    ROBOFLOW_API_KEY = args.api_key or get_env_var('ROBOFLOW_API_KEY')

    # Get dataset path from command line arguments or environment variables
    DATASET_PATH = args.dataset_path or get_env_var('DATASET_PATH')
    
    # Initialize Roboflow
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    
    # Get project name from command line arguments only, or ask interactively if not provided
    workspace = rf.workspace()
    PROJECT_NAME = select_project(workspace, args.project)
    project = workspace.project(PROJECT_NAME)
    
    # If the project string contains a slash, extract only the project slug
    if "/" in PROJECT_NAME:
        PROJECT_NAME = PROJECT_NAME.split("/")[-1]
    project = rf.workspace().project(PROJECT_NAME)
    
    # Get dataset version from command line arguments or environment variables
    if not args.version:
        print("No dataset version provided, will select the highest version")
        versions = project.versions()
        DATASET_VERSION = max(version.version for version in versions)
    else:
        DATASET_VERSION = args.version

    # Get dataset format from command line arguments
    print(f"Dataset format selected: {args.dataset_format}")  # Print the dataset format

    # Download the dataset
    dataset = project.version(DATASET_VERSION).download(model_format=args.dataset_format)
    print(f"Dataset downloaded successfully to: {dataset.location}")

    # Create the destination folder if it doesn't exist
    destination_folder = Path(DATASET_PATH).resolve()  # Use pathlib for the destination folder
    destination_folder.mkdir(parents=True, exist_ok=True)  # Create the folder if it doesn't exist

    # Get the dataset name from the downloaded dataset
    dataset_name = Path(dataset.location).name
    dataset_path = destination_folder / dataset_name  # Full path to the dataset folder

    # Remove the folder if it exists
    if dataset_path.exists() and dataset_path.is_dir():
        shutil.rmtree(dataset_path)  # Remove the existing dataset folder

    # Move the dataset to the destination folder
    shutil.move(dataset.location, dataset_path)  # Move the dataset
    print(f"Dataset moved to: {dataset_path}")

    return str(dataset_path)  # Return the new location as a string

if __name__ == "__main__":
    main()