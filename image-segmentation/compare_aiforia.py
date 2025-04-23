import pandas as pd
import json
import argparse
from shapely.geometry import Point, mapping
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import minimize, linear_sum_assignment
import cv2
import matplotlib.pyplot as plt
import os

def extract_centers(geojson_path):
    """
    Extract center coordinates of bounding boxes from a GeoJSON file.

    Args:
        geojson_path (str): Path to the input GeoJSON file.
    
    Returns:
        list: A list of tuples containing the center coordinates (x, y).
    """
    # Load the GeoJSON file
    with open(geojson_path) as f:
        geojson_data = json.load(f)

    centers = []

    # Iterate over the features in the GeoJSON
    for feature in geojson_data['features']:
        # Get the bounding box coordinates
        bbox = feature['properties'].get('bbox')  # Assuming bbox is stored in properties
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            # Calculate the center coordinates
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            centers.append((center_x, center_y))
        else:
            print(f"Invalid bounding box for feature {feature['id']}. Skipping.")

    return centers

def align_point_clouds(pc1, pc2):
    tree = cKDTree(pc2)

    def cost(t):
        translated = pc1 + t
        dists, _ = tree.query(translated)
        return np.mean(dists**2)

    result = minimize(cost, x0=np.zeros(2))  # Only (tx, ty)
    return result.x  # Optimal translation vector

def match_points(pc1, pc2, threshold=20):
    dist_matrix = cdist(pc1, pc2)
    dist_matrix[dist_matrix > threshold] = 1e6  # Mask out long distances

    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    matches = [(i, j) for i, j in zip(row_ind, col_ind) if dist_matrix[i, j] < 1e6]

    matched_pc1 = set(i for i, _ in matches)
    matched_pc2 = set(j for _, j in matches)
    unmatched_pc1 = list(set(range(len(pc1))) - matched_pc1)
    unmatched_pc2 = list(set(range(len(pc2))) - matched_pc2)

    return matches, unmatched_pc1, unmatched_pc2

def read_csv(csv_path, pixel_size):
    """
    Read the CSV file and scale the coordinates.

    Args:
        csv_path (str): Path to the input CSV file.
        pixel_size (float): The pixel size to convert coordinates.

    Returns:
        DataFrame: A DataFrame containing the scaled coordinates.
    """
    # Read the CSV file
    df = pd.read_csv(csv_path, sep='\t')  # Use '\t' as the separator for tab-separated values

    # Check if required columns exist
    if 'Object center X (μm)' not in df.columns or 'Object center Y (μm)' not in df.columns:
        raise ValueError("CSV file must contain 'Object center X (μm)' and 'Object center Y (μm)' columns.")

    # Scale the coordinates by pixel size
    df['Object center X (μm)'] = df['Object center X (μm)'] / pixel_size  # In pixels
    df['Object center Y (μm)'] = df['Object center Y (μm)'] / pixel_size  # In pixels

    return df

def save_geojson(geojson_path, df, num_matches, num_unmatched1, num_unmatched2):
    """
    Save features to a GeoJSON file.

    Args:
        geojson_path (str): Path to the output GeoJSON file.
        df (DataFrame): DataFrame containing the features to save.
        num_matches (int): Number of matches found.
        num_unmatched1 (int): Number of unmatched centers in centers1.
        num_unmatched2 (int): Number of unmatched centers in centers2.
    """
    # Create a list to hold GeoJSON features
    features = []

    # Iterate over the rows in the DataFrame
    for _, row in df.iterrows():
        # Get the object center coordinates
        x = row['Object center X (μm)']
        y = row['Object center Y (μm)']
        
        # Create a point geometry
        point = Point(x, y)

        # Create a GeoJSON feature
        feature = {
            "type": "Feature",
            "geometry": mapping(point),
            "properties": {
                "class_label": row['Class label'],
                "class_confidence": row['Class confidence (%)'],
                "area": row['Area (μm²)']
            }
        }
        features.append(feature)

    # Create the GeoJSON structure
    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "num_matches": num_matches,
            "num_unmatched_aifoira": num_unmatched1,
            "num_unmatched_multipark": num_unmatched2
        }
    }

    # Ensure the directory exists before writing the GeoJSON file
    geojson_dir = os.path.dirname(geojson_path)
    if not os.path.exists(geojson_dir):
        os.makedirs(geojson_dir)  # Create the directory if it doesn't exist

    # Write the GeoJSON to a file
    with open(geojson_path, 'w') as f:
        json.dump(geojson, f, indent=2)

    print(f"GeoJSON saved to {geojson_path}")

def plot_centers(image_path, centers, centers2, save_image_path=None, matches=None):
    """
    Plot the centers on the image or save the image to a file.

    Args:
        image_path (str): Path to the image file.
        centers (list): List of tuples containing center coordinates (x, y).
        centers2 (list): List of tuples containing second set of center coordinates (x, y).
        save_image_path (str, optional): Path to save the plotted image. If None, the image will be displayed.
        matches (list, optional): List of matched indices to draw lines between.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # Draw the centers on the image
    for center in centers:
        cv2.circle(image, (int(center[0]), int(center[1])), radius=5, color=(0, 0, 255), thickness=-1)  # Red dots for centers (Aiforia)
    for center in centers2:
        cv2.circle(image, (int(center[0]), int(center[1])), radius=5, color=(0, 255, 0), thickness=-1)  # Green dots for centers2 (Multipark)

    # Draw lines connecting matched centers
    if matches is not None:
        for match in matches:
            center1 = centers[match[0]]
            center2 = centers2[match[1]]
            cv2.line(image, (int(center1[0]), int(center1[1])), (int(center2[0]), int(center2[1])), color=(255, 0, 0), thickness=2)  # Yellow lines for matches

    # Create a legend
    legend_x, legend_y = 10, 10  # Starting position for the legend
    legend_width, legend_height = 200, 60  # Size of the legend box
    cv2.rectangle(image, (legend_x, legend_y), (legend_x + legend_width, legend_y + legend_height), (255, 255, 255), -1)  # White background for legend

    # Draw legend items
    cv2.rectangle(image, (legend_x + 10, legend_y + 10), (legend_x + 30, legend_y + 30), (0, 0, 255), -1)  # Red box for Aiforia
    cv2.putText(image, 'Aiforia', (legend_x + 40, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)  # Text for Aiforia

    cv2.rectangle(image, (legend_x + 10, legend_y + 40), (legend_x + 30, legend_y + 60), (0, 255, 0), -1)  # Green box for Multipark
    cv2.putText(image, 'Multipark', (legend_x + 40, legend_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)  # Text for Multipark

    # Save or show the image
    if save_image_path:
        cv2.imwrite(save_image_path, image)  # Save the image with drawn centers
        print(f"Image saved to {save_image_path}")
    else:
        cv2.imshow("Detected Centers", image)  # Display the image
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()  # Close the window

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert CSV object center coordinates to GeoJSON and plot centers.')
    parser.add_argument('--aiforia-csv', type=str, required=True, help='Path to the Aiforia detections CSV file.')
    parser.add_argument('--multipark-geojson', type=str, required=True, help='Path to the Multipark detections GeoJSON file.')
    parser.add_argument('--output-path', type=str, required=True, help='Path to the output directory for GeoJSON and images.')
    parser.add_argument('--pixel-size', type=float, required=True, help='Pixel size to convert coordinates.')
    parser.add_argument('--image', type=str, help='Path to the image file to plot centers on.')
    parser.add_argument('--save-geojson', action='store_true', help='Flag to save the GeoJSON file.')
    parser.add_argument('--save-image', action='store_true', help='Flag to save the plotted image instead of displaying it.')

    args = parser.parse_args()

    # Read the Aiforia CSV and scale coordinates
    df = read_csv(args.aiforia_csv, args.pixel_size)

    # Align
    centers1 = np.column_stack((df['Object center X (μm)'], df['Object center Y (μm)']))
    centers2 = extract_centers(args.multipark_geojson)  # Extract centers from detections.
    tx, ty = align_point_clouds(centers1, centers2)
    
    # Update the DataFrame with translated coordinates
    df['Object center X (μm)'] = df['Object center X (μm)'] + tx
    df['Object center Y (μm)'] = df['Object center Y (μm)'] + ty
    
    # Correctly translate centers2
    centers1_translated = centers1 + np.array([tx, ty])  # Translate centers1 by (tx, ty)

    # Call match_point_clouds
    matches, unmatched1, unmatched2 = match_points(centers1_translated, centers2, threshold=40)  # Example threshold of 20 pixels

    # Print the number of matches and unmatched centers
    print(f"Number of matches: {len(matches)}")
    print(f"Number of unmatched Aiforia detections: {len(unmatched1)}")
    print(f"Number of unmatched Multipark detections: {len(unmatched2)}")

    # Save GeoJSON if the flag is set
    if args.save_geojson:
        base_name = os.path.splitext(os.path.basename(args.image))[0]  # Get base name from the input image
        geojson_path = os.path.join(args.output_path, f"{base_name}.geojson")
        save_geojson(geojson_path, df, len(matches), len(unmatched1), len(unmatched2))

    # If an image path is provided, plot the centers
    if args.image:
        if args.save_image:
            save_image_path = os.path.join(args.output_path, f"{base_name}_centers.png")  # Save image with base name
        else:
            save_image_path = None
        
        plot_centers(args.image, centers1_translated, centers2, save_image_path, matches)  # Pass the save_image path

if __name__ == "__main__":
    main()