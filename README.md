# multipark-aiml

Multipark AI/ML Platform
=======================

This repository contains tools and scripts for image segmentation, model training, inference, and dataset management using YOLO and Roboflow.

## Features
- Download datasets from Roboflow interactively or via CLI
- Train YOLO segmentation models
- Run inference on images or folders (with GUI and CLI options)
- Deploy trained models to Roboflow
- Create and use polygon masks for region-of-interest inference
- Compare segmentation results with Aiforia outputs

## Requirements
- Python 3.8+ (tested with 3.11)
- See `image-segmentation/requirements.txt` for Python dependencies

## Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/your-org/multipark-aiml.git
   cd multipark-aiml/image-segmentation
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Create a `.env` file in `image-segmentation/` with your Roboflow API key and paths:
   ```ini
   ROBOFLOW_API_KEY="your_roboflow_api_key"
   DATASET_PATH="./datasets"
   MODEL_PATH="./models"
   ```

## Usage

### Download a Dataset
```
python download_dataset.py --api-key <API_KEY> [--project <project_slug>] [--version <version>] [--dataset-format yolov11] [--dataset-path <path>]
```
If `--project` is not provided, you will be prompted to select one interactively.

### Train a YOLO Model
```
python train_yolo.py --dataset-dir <dataset_dir> [--epochs 100] [--imgsz 640] [--batch -1] [--device <device>] [--model yolo11n-seg]
```

### Run Inference (CLI)
```
python infer_yolo.py --model <model_path> --input-path <image_or_folder> --output-path <output_folder> [options]
```
Key options:
- `--no-colors`: Convert output to grayscale
- `--sliding-window`: Use sliding window for large images
- `--save-image`: Save visualization
- `--save-geojson`: Save results as GeoJSON
- `--use-geojson-mask`: Use a polygon mask for ROI

### Run Inference (GUI)
```
python gui_infer_yolo.py
```
A graphical interface will open for model selection and inference. (Tested on Mac, tested and not working on Linux CentOS.)

### Deploy a Model to Roboflow
```
python deploy_model.py --model <model_path> --api-key <API_KEY> [--project <project_slug>] [--version <version>]
```
If `--project` is not provided, you will be prompted to select one interactively.

### Create a Polygon Mask
```
python create_mask.py --image <image_path> [--output <output_geojson>]
```
Interactively draw a polygon on the image to create a GeoJSON mask.

### Compare with Aiforia Results
```
python compare_aiforia.py --aiforia-csv <csv> --multipark-geojson <geojson> --output-path <dir> --pixel-size <size> [--image <img>] [--save-geojson] [--save-image]
```

## Best Practices
- Keep your `.env` file in `image-segmentation/` and add it to `.gitignore`.
- Do not commit sensitive information (API keys, etc.) to version control.
- Use a virtual environment for Python dependencies.

## License
See [LICENSE](LICENSE) for details.

## Contact
For questions or contributions, please open an issue or pull request.
