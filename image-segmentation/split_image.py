import os
import numpy as np
from PIL import Image
import argparse
from pathlib import Path

def split_image(image_path, output_dir, tile_size=640, overlap=0):
    """
    Split an image into tiles of specified size with optional overlap
    
    Args:
        image_path (str): Path to the input image
        output_dir (str): Directory to save the tiles
        tile_size (int): Size of each tile (both width and height)
        overlap (int): Number of pixels to overlap between tiles
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the image
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Get image dimensions
    height, width = img_array.shape[:2]
    
    # Calculate number of tiles
    n_tiles_h = (height - overlap) // (tile_size - overlap) + 1
    n_tiles_w = (width - overlap) // (tile_size - overlap) + 1
    
    print(f"Splitting image '{image_path}' into {n_tiles_h}x{n_tiles_w} tiles...")
    
    # Split the image into tiles
    for i in range(n_tiles_h):
        for j in range(n_tiles_w):
            # Calculate tile coordinates
            y_start = i * (tile_size - overlap)
            y_end = min(y_start + tile_size, height)
            x_start = j * (tile_size - overlap)
            x_end = min(x_start + tile_size, width)
            
            # Extract tile
            tile = img_array[y_start:y_end, x_start:x_end]
            
            # Create output filename
            output_path = os.path.join(output_dir, f"tile_{os.path.basename(image_path).split('.')[0]}_{i}_{j}.png")
            
            # Save tile
            Image.fromarray(tile).save(output_path)
    
    print(f"Tiles saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Split an image into tiles')
    parser.add_argument('input_path', help='Path to the input image or directory')
    parser.add_argument('--output-dir', help='Directory to save the tiles (default: input filename in input directory)')
    parser.add_argument('--tile-size', type=int, default=640, help='Size of each tile (default: 640)')
    parser.add_argument('--overlap', type=int, default=0, help='Number of pixels to overlap between tiles (default: 0)')
    
    args = parser.parse_args()
    
    # Check if input path exists
    if not os.path.exists(args.input_path):
        print(f"Error: Input path '{args.input_path}' does not exist")
        return
    
    # Check if input path is a directory or a file
    if os.path.isdir(args.input_path):
        # Iterate over all image files in the directory
        for filename in os.listdir(args.input_path):
            if filename.lower().endswith(('.png', '.tiff', '.tif', '.jpg', '.jpeg')):
                image_path = os.path.join(args.input_path, filename)
                # Set default output directory
                if args.output_dir is None:
                    args.output_dir = str(Path(args.input_path) / Path(filename).stem)
                split_image(image_path, args.output_dir, args.tile_size, args.overlap)
    else:
        # Process a single image file
        if not args.input_path.lower().endswith(('.png', '.tiff', '.tif', '.jpg', '.jpeg')):
            print("Error: Input file must be a PNG, TIFF, or JPEG image")
            return
        
        # Set default output directory
        if args.output_dir is None:
            input_path = Path(args.input_path)
            args.output_dir = str(input_path.parent / input_path.stem)
        
        split_image(args.input_path, args.output_dir, args.tile_size, args.overlap)

if __name__ == "__main__":
    main() 