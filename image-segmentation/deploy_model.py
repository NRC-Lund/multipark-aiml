import argparse
from roboflow import Roboflow
from config import get_env_var, select_project


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Deploy a YOLO model to Roboflow')
    parser.add_argument('--project', type=str, required=True, help='Roboflow project name')
    parser.add_argument('--version', type=int, help='Roboflow dataset version')
    parser.add_argument('--api-key', type=str, help='Roboflow API key')
    parser.add_argument('--model', type=str, required=True, help='Path to the YOLO model directory')
    args = parser.parse_args()

    # Get API key from command line arguments or environment variables
    ROBOFLOW_API_KEY = args.api_key or get_env_var('ROBOFLOW_API_KEY')

    # Initialize Roboflow
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    workspace = rf.workspace()

    # Get project name from command line arguments only, or ask interactively if not provided
    PROJECT_NAME = select_project(workspace, args.project)
    project = workspace.project(PROJECT_NAME)

    # Get dataset path from command line arguments or environment variables
    if not args.version:
        print("No dataset version provided, will select the highest version")
        versions = project.versions()
        DATASET_VERSION = max(version.version for version in versions)
    else:
        DATASET_VERSION = args.version

    # Deploy the model
    print(f"Deploying model to version {DATASET_VERSION} of project {PROJECT_NAME}")
    version = project.version(DATASET_VERSION)
    version.deploy(model_type="yolov11-seg",
                   model_path=args.model,
                   filename="weights/best.pt")

if __name__ == "__main__":
    main()