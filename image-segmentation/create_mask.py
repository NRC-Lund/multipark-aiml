import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon, mapping
import cv2
import os  # Import os to handle directory operations

# Global variable to store polygon vertices
vertices = []

def onclick(event):
    """Capture mouse click events to define polygon vertices."""
    global vertices
    if event.xdata is not None and event.ydata is not None:
        vertices.append((event.xdata, event.ydata))
        plt.scatter(event.xdata, event.ydata, color='red')  # Mark the vertex
        plt.draw()

def on_key(event):
    """Handle key press events."""
    if event.key == 'enter':
        plt.close()  # Close the plot when Enter is pressed

def create_geojson_mask(vertices, output_path):
    """Create a GeoJSON mask from the given vertices."""
    polygon = Polygon(vertices)

    # Create GeoJSON structure
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": mapping(polygon),
                "properties": {}
            }
        ]
    }

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)  # Create the directory if it doesn't exist

    # Write the GeoJSON to a file
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)
    
    print(f"GeoJSON mask saved to {output_path}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Create a GeoJSON mask by drawing a polygon on an image.')
    parser.add_argument('--image', type=str, required=True, help='Path to the image file.')
    parser.add_argument('--output', type=str, help='Output path for the GeoJSON file.')  # Make output optional

    args = parser.parse_args()

    # Load the image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image at {args.image}")
        return

    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Set up the plot
    fig, ax = plt.subplots()
    ax.imshow(image_rgb)
    plt.title("Click to define polygon vertices. Press Enter to finish.")

    # Connect the click event to the onclick function
    cid_click = fig.canvas.mpl_connect('button_press_event', onclick)
    cid_key = fig.canvas.mpl_connect('key_press_event', on_key)  # Connect key press event

    # Show the plot
    plt.show()

    # Disconnect events after closing the plot
    plt.disconnect(cid_click)
    plt.disconnect(cid_key)

    # Create the GeoJSON mask
    if len(vertices) < 3:
        print("Error: At least three vertices are required to create a polygon.")
        return

    # Determine output path
    if args.output is None:
        # Create output path based on input image path
        base_name = os.path.splitext(args.image)[0]  # Get the base name without extension
        args.output = f"{base_name}.geojson"  # Set output path to base name with .geojson extension

    create_geojson_mask(vertices, args.output)

if __name__ == "__main__":
    main()