from roboflow import Roboflow
from ultralytics import YOLO
from pathlib import Path
from dotenv import load_dotenv
import argparse
import torch

def get_device(args_device):
    """Return the device string for Ultralytics YOLO training."""
    if args_device:
        return args_device
    if torch.cuda.is_available():
        return '0'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train YOLO segmentation model')
    parser.add_argument('--dataset-dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=-1, help='Batch size')
    parser.add_argument('--device', type=str, help='Device to use for training')
    parser.add_argument('--model', type=str, default='yolo11n-seg', help='Model file name')
    args = parser.parse_args()

    device = get_device(args.device)
    print("Using device:", device)

    # Load model
    print("Initializing model...")
    model = YOLO(args.model)  # Load YOLO segmentation model
    
    # Get data.yaml path
    data_yaml_path = Path(args.dataset_dir) / "data.yaml"
    data_yaml_path = data_yaml_path.resolve()   # Path must be absolute

    # Train model
    print("Starting training...")
    results = model.train(
        data=data_yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device
    )

    return results.save_dir

if __name__ == "__main__":
    main()