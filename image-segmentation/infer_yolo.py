from ultralytics import YOLO
import cv2
import numpy as np
import os
from dataclasses import dataclass
import argparse
import json
from shapely.geometry import Polygon

@dataclass
class VisualizationConfig:
    """Configuration for visualization options"""
    draw_boxes: bool = False
    draw_labels: bool = False
    draw_contours: bool = True
    draw_masks: bool = True  # For filled masks
    box_color: tuple = (0, 255, 0)  # Green
    contour_color: tuple = (0, 0, 255)  # Red
    mask_alpha: float = 0.3  # Transparency for filled masks
    no_colors: bool = False  # Whether to convert the image to grayscale
    contour_thickness: int = 2  # Thickness of contour lines

@dataclass
class SlidingWindowConfig:
    """Configuration for sliding window inference"""
    window_size: int = 640  # Size of each window
    overlap: int = 128  # Overlap between windows
    conf_threshold: float = 0.25  # Confidence threshold for detections

def visualize_detections(image, results, config: VisualizationConfig = None):
    """
    Visualize detection results from YOLO model
    
    Args:
        image: Input image
        results: List of dictionaries containing detection results
        config: VisualizationConfig object controlling what to draw
    """
    if config is None:
        config = VisualizationConfig()
    
    # Create a copy of the image to draw on
    img = image.copy()
    
    # Convert to grayscale if no_colors is True
    if config.no_colors:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert back to BGR for drawing
    
    # Draw filled masks first (if enabled) so they appear behind other elements
    if config.draw_masks:
        for result in results:
            if result['mask'] is not None:
                # Create a colored mask for this instance
                color = (
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                    np.random.randint(0, 255)
                )
                mask_img = np.zeros_like(img)
                binary_mask = (result['mask'] > 0.5).astype(np.uint8)
                mask_img[binary_mask > 0] = color
                
                # Blend the mask with the image
                cv2.addWeighted(
                    img,
                    1,
                    mask_img,
                    config.mask_alpha,
                    0,
                    img
                )
    
    # Draw each detection
    for result in results:
        # Get box coordinates
        x1, y1, x2, y2 = result['box']
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Get confidence and class
        conf = result['confidence']
        cls_id = result['class_id']
        cls_name = result['class_name']  # Use the class name from results
        
        # Draw rectangle if enabled
        if config.draw_boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), config.box_color, 2)
        
        # Add label with class name and confidence if enabled
        if config.draw_labels:
            label = f'{cls_name} {conf:.2f}'
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.box_color, 2)
    
    # Draw contours if enabled
    if config.draw_contours:
        for result in results:
            if result['polygon'] is not None:
                # Convert polygon to contour format
                contour = result['polygon'].astype(np.int32)
                # Reshape for polylines
                contour = contour.reshape((-1, 1, 2))
                cv2.polylines(img, [contour], True, config.contour_color, config.contour_thickness)
    
    return img

def polygon_iou(p1, p2):
    if not p1.is_valid or not p2.is_valid:
        return 0.0
    inter = p1.intersection(p2).area
    union = p1.union(p2).area
    return inter / union if union > 0 else 0.0

def polygon_nms(results, iou_threshold=0.5):
    """
    Applies Non-Maximum Suppression to polygon-based detection results.
    Each result must have a 'polygon', 'confidence', and 'class_id'.
    """
    # Sort by confidence descending
    results = sorted(results, key=lambda r: r['confidence'], reverse=True)
    keep = []
    
    while results:
        current = results.pop(0)
        keep.append(current)
        
        filtered = []
        for other in results:
            # Only compare same class
            if current['class_id'] != other['class_id']:
                filtered.append(other)
                continue
            
            # Fallback to box IoU if no polygon
            if current['polygon'] is None or other['polygon'] is None:
                filtered.append(other)
                continue

            iou = polygon_iou(Polygon(current['polygon']), Polygon(other['polygon']))
            if iou < iou_threshold:
                filtered.append(other)

        results = filtered
    return keep

def run_sliding_window_inference(model_path, image_path, window_config: SlidingWindowConfig, vis_config: VisualizationConfig = None):
    """
    Run inference using a sliding window approach for large images
    
    Args:
        model_path (str): Path to the YOLO model weights
        image_path (str): Path to the image to process
        window_config: SlidingWindowConfig object with window parameters
        vis_config: VisualizationConfig object controlling what to draw
    
    Returns:
        tuple: (image with detections, combined results, contours, confidences)
    """
    # Load the model
    model = YOLO(model_path)
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Could not load image at {image_path}")
    
    height, width = image.shape[:2]
    window_size = window_config.window_size
    overlap = window_config.overlap
    stride = window_size - overlap
    
    print(f"Running sliding window inference on image {width}x{height} with window size {window_size}, overlap {overlap}, and stride {stride}")

    # Initialize the output image and mask
    output_image = image.copy()
    all_results = []
    
    # Process each window
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            # Calculate window boundaries
            x1, y1 = x, y
            x2 = min(x + window_size, width)
            y2 = min(y + window_size, height)
            
            # Extract the window
            window = image[y1:y2, x1:x2]
            
            # Skip if window is empty
            if window.size == 0:
                continue
            
            # If window is not full size, pad it
            if window.shape[0] < window_size or window.shape[1] < window_size:
                padded_window = np.zeros((window_size, window_size, 3), dtype=np.uint8)
                padded_window[:window.shape[0], :window.shape[1]] = window
                window = padded_window
            
            # Run inference on the window
            results = model(
                window,
                conf=window_config.conf_threshold,
                verbose=False
            )
            
            # Convert results to dictionary format
            window_results = convert_yolo_to_dict(results[0])
            
            # Process each detection in the window
            for result in window_results:
                # Apply window offset to box coordinates
                result['box'][0] += x1  # Add x offset
                result['box'][1] += y1  # Add y offset
                result['box'][2] += x1  # Add x offset
                result['box'][3] += y1  # Add y offset
                
                # Apply window offset to polygon coordinates if available
                if result['polygon'] is not None:
                    result['polygon'][:, 0] += x1  # Add x offset
                    result['polygon'][:, 1] += y1  # Add y offset
                
                # Handle mask if available
                if result['mask'] is not None:
                    # Get the actual mask dimensions
                    mask = result['mask']
                    mask_height, mask_width = mask.shape[:2]
                    
                    # Calculate the valid region in the full image
                    valid_height = min(mask_height, y2 - y1)
                    valid_width = min(mask_width, x2 - x1)
                    
                    # Create a full-size mask
                    full_mask = np.zeros((height, width), dtype=np.uint8)
                    
                    # Copy only the valid portion of the mask
                    if valid_height > 0 and valid_width > 0:
                        full_mask[y1:y1+valid_height, x1:x1+valid_width] = mask[:valid_height, :valid_width]
                    
                    # Update the result with the full-size mask
                    result['mask'] = full_mask
                
                # Append the adjusted result
                all_results.append(result)
    
    # Remove duplicate detections
    fused_results = polygon_nms(all_results, iou_threshold=0.4)

    # Visualize results
    img_with_detections = visualize_detections(image, fused_results, vis_config)
    
    return img_with_detections, fused_results

def convert_yolo_to_dict(result):
    """
    Convert YOLO results to a list of dictionaries with boxes, confidences, classes, polygons, and masks
    
    Args:
        result: YOLO prediction result for a single image
    
    Returns:
        list: List of dictionaries containing detection results
    """
    # Convert boxes to numpy arrays
    boxes = result.boxes
    boxes_xyxy = boxes.xyxy.cpu().numpy()
    boxes_conf = boxes.conf.cpu().numpy()
    boxes_cls = boxes.cls.cpu().numpy()
    
    # Get class names from the model
    class_names = result.names if hasattr(result, 'names') else None
    
    # Convert masks to numpy arrays if available
    masks_data = None
    masks_xy = None
    if hasattr(result, 'masks') and result.masks is not None:
        masks_data = result.masks.data.cpu().numpy()
        if hasattr(result.masks, 'xy'):
            masks_xy = result.masks.xy
    
    # Create results dictionary
    results_dict = []
    for i, (box, conf, cls_id) in enumerate(zip(boxes_xyxy, boxes_conf, boxes_cls)):
        # Get mask and polygon for this detection if available
        mask = masks_data[i] if masks_data is not None else None
        poly = masks_xy[i] if masks_xy is not None else None
        
        # Get class name if available, otherwise use class_id
        cls_name = class_names[int(cls_id)] if class_names is not None else f"class_{int(cls_id)}"
        
        results_dict.append({
            "box": box,                # shape: (4,)
            "confidence": float(conf),
            "class_id": int(cls_id),
            "class_name": cls_name,    # Name of the class
            "polygon": poly,           # shape: (N, 2)
            "mask": mask               # shape: (H, W) or None
        })
    
    return results_dict

def run_inference(model_path, image_path, conf_threshold=0.25, vis_config: VisualizationConfig = None):
    """
    Run inference using a YOLO model
    
    Args:
        model_path (str): Path to the YOLO model weights
        image_path (str): Path to the image to process
        conf_threshold (float): Confidence threshold for detections
        vis_config: VisualizationConfig object controlling what to draw
    
    Returns:
        tuple: (image with detections, results)
    """
    # Load the model
    model = YOLO(model_path)
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Could not load image at {image_path}")
    
    # Run inference
    results = model(
        image,
        conf=conf_threshold,
        verbose=False
    )
    
    # Convert results to dictionary format
    results_dict = convert_yolo_to_dict(results[0])
    
    # Visualize results
    img_with_detections = visualize_detections(image, results_dict, vis_config)
    
    return img_with_detections, results_dict

def convert_to_geojson(results, image_shape, contours=None, confidence_mask=None):
    """
    Convert detection results to GeoJSON format
    
    Args:
        results: List of dictionaries containing detection results
        image_shape: Tuple of (height, width) of the original image
        contours: List of contours from sliding window inference (deprecated)
        confidence_mask: Confidence values for each pixel in the mask (deprecated)
    
    Returns:
        dict: Results in GeoJSON format
    """
    height, width = image_shape[:2]
    
    # Initialize GeoJSON structure with image dimensions and count in properties
    geojson = {
        "type": "FeatureCollection",
        "properties": {
            "image_width": width,
            "image_height": height,
            "num_objects": len(results)
        },
        "features": []
    }
    
    # Process each detection
    for result in results:
        # Get box coordinates
        x1, y1, x2, y2 = result['box']
        
        # Get confidence and class
        conf = result['confidence']
        cls_id = result['class_id']
        cls_name = result['class_name']  # Get the class name
        
        # Get polygon coordinates if available
        coordinates = []
        area = 0.0
        if result['polygon'] is not None:
            # Convert polygon points to GeoJSON format
            coordinates = result['polygon'].tolist()
            # Close the polygon by adding the first point at the end
            if coordinates and coordinates[0] != coordinates[-1]:
                coordinates.append(coordinates[0])
            # Calculate area from polygon
            area = float(cv2.contourArea(np.array(coordinates, dtype=np.int32)))
        elif result['mask'] is not None:
            # If we have a mask but no polygon, create coordinates from mask
            mask = result['mask']
            y_coords, x_coords = np.where(mask > 0.5)
            for x, y in zip(x_coords, y_coords):
                coordinates.append([float(x), float(y)])
            # Calculate area from mask
            area = float(np.sum(mask > 0.5))
        
        # Create GeoJSON feature
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coordinates] if coordinates else None
            },
            "properties": {
                "confidence": conf,
                "area": area,
                "class_id": cls_id,
                "class_name": cls_name,  # Add class name to properties
                "bbox": [float(x1), float(y1), float(x2), float(y2)]
            }
        }
        geojson["features"].append(feature)
    
    return geojson

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run YOLO inference on an image or directory of images')
    parser.add_argument('--model', type=str, required=True, help='Path to the YOLO model weights')
    parser.add_argument('--input-path', type=str, required=True, help='Path to input image or directory of images')
    parser.add_argument('--output-path', type=str, required=True, help='Path to output directory')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for detections')
    parser.add_argument('--sliding-window', action='store_true', help='Use sliding window inference')
    parser.add_argument('--window-size', type=int, default=640, help='Size of each window for sliding window inference')
    parser.add_argument('--overlap', type=int, default=128, help='Overlap between windows for sliding window inference')
    parser.add_argument('--no-boxes', action='store_true', help='Do not draw bounding boxes')
    parser.add_argument('--no-labels', action='store_true', help='Do not draw labels')
    parser.add_argument('--no-contours', action='store_true', help='Do not draw contours')
    parser.add_argument('--no-masks', action='store_true', help='Do not draw filled masks')
    parser.add_argument('--no-colors', action='store_true', help='Convert input image to grayscale')
    parser.add_argument('--no-display', action='store_true', help='Do not display the image on screen')
    parser.add_argument('--box-color', type=str, default='0,255,0', help='Color for bounding boxes (B,G,R)')
    parser.add_argument('--contour-color', type=str, default='0,0,255', help='Color for contours (B,G,R)')
    parser.add_argument('--contour-thickness', type=int, default=1, help='Thickness of contour lines')
    parser.add_argument('--mask-alpha', type=float, default=0.3, help='Transparency for filled masks')
    parser.add_argument('--save-image', action='store_true', help='Save visualization image')
    parser.add_argument('--save-geojson', action='store_true', help='Save results in GeoJSON format')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    
    # Create visualization config
    vis_config = VisualizationConfig(
        draw_boxes=not args.no_boxes,
        draw_labels=not args.no_labels,
        draw_contours=not args.no_contours,
        draw_masks=not args.no_masks,
        box_color=tuple(map(int, args.box_color.split(','))),
        contour_color=tuple(map(int, args.contour_color.split(','))),
        mask_alpha=args.mask_alpha,
        no_colors=args.no_colors,
        contour_thickness=args.contour_thickness
    )
    
    # Get list of input images
    if os.path.isfile(args.input_path):
        input_images = [args.input_path]
    elif os.path.isdir(args.input_path):
        input_images = [
            os.path.join(args.input_path, f) for f in os.listdir(args.input_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))
        ]
    else:
        print(f"Error: Input path {args.input_path} does not exist")
        return
    
    if not input_images:
        print("Error: No valid input images found")
        return
    
    try:
        for input_image in input_images:
            print(f"\nProcessing {input_image}...")
            
            # Get base filename without extension
            base_name = os.path.splitext(os.path.basename(input_image))[0]
            
            if args.sliding_window:
                # Create sliding window config
                window_config = SlidingWindowConfig(
                    window_size=args.window_size,
                    overlap=args.overlap,
                    conf_threshold=args.conf
                )
                
                # Run sliding window inference
                img_with_detections, results = run_sliding_window_inference(
                    model_path=args.model,
                    image_path=input_image,
                    window_config=window_config,
                    vis_config=vis_config
                )
                
                # Load the image for shape information
                image = cv2.imread(input_image)
                if image is None:
                    raise ValueError(f"Error: Could not load image at {input_image}")
                
                contours = None
                confidence_mask = None
                
            else:
                # Run regular inference
                img_with_detections, results = run_inference(
                    model_path=args.model,
                    image_path=input_image,
                    conf_threshold=args.conf,
                    vis_config=vis_config
                )
                image = cv2.imread(input_image)
                contours = None
                confidence_mask = None
            
            # Save results based on flags
            if args.save_geojson:
                geojson_path = os.path.join(args.output_path, f"{base_name}.geojson")
                geojson_data = convert_to_geojson(results, image.shape, contours, confidence_mask)
                with open(geojson_path, 'w') as f:
                    json.dump(geojson_data, f, indent=2)
                print(f"GeoJSON results saved to {geojson_path}")
            
            if args.save_image:
                # Get the file extension from the input image
                _, ext = os.path.splitext(input_image)
                image_path = os.path.join(args.output_path, f"{base_name}_detections{ext}")
                cv2.imwrite(image_path, img_with_detections)
                print(f"Visualization saved to {image_path}")
            
            # Display the image if not disabled
            if not args.no_display:
                window_name = f'Detection Results - {os.path.basename(input_image)}'
                cv2.imshow(window_name, img_with_detections)
                cv2.waitKey(0)
                cv2.destroyAllWindows()  # Close all windows
            
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise  # This will show the full traceback

if __name__ == "__main__":
    main() 