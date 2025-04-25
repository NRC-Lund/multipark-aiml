# Minimal Flask web app for infer_yolo
import os
import sys
from flask import Flask, request, render_template, send_from_directory, redirect, url_for, session, jsonify
import subprocess
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import cv2
import numpy as np
import json
import datetime
from dotenv import load_dotenv
from config import DevelopmentConfig, ProductionConfig, setup_logging
import logging

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

app = Flask(__name__)

# Set secret key from environment variable
app.secret_key = os.getenv('SECRET_KEY', 'your_secret_key')  # Fallback for safety

# Choose config based on environment variable
if os.getenv('FLASK_ENV') == 'production':
    app.config.from_object(ProductionConfig)
    setup_logging(debug=False)
else:
    app.config.from_object(DevelopmentConfig)
    setup_logging(debug=True)

PYTHON_EXECUTABLE = app.config.get('PYTHON_EXECUTABLE', sys.executable)
UPLOAD_FOLDER = app.config['UPLOAD_FOLDER']
OUTPUT_FOLDER = app.config['OUTPUT_FOLDER']
MODEL_FOLDER = app.config['MODEL_FOLDER']
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}

logging.info(f"Upload folder: {UPLOAD_FOLDER}")
logging.info(f"Output folder: {OUTPUT_FOLDER}")
logging.info(f"Model folder: {MODEL_FOLDER}")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_default_model_path():
    return os.path.join(MODEL_FOLDER, 'th-stained-dopamine-neurons-v3-medium.pt')

def get_model_paths():
    return [os.path.join(MODEL_FOLDER, fname) for fname in os.listdir(MODEL_FOLDER) if fname.endswith('.pt')]

def get_uploaded_files():
    files = []
    for fname in os.listdir(UPLOAD_FOLDER):
        if allowed_file(fname):
            files.append(fname)
    return sorted(files)

@app.route('/', methods=['GET', 'POST'])
def upload_and_infer():
    uploaded_files = get_uploaded_files()
    uploaded_img = session.get('uploaded_img')
    # Inference options
    model_path = request.form.get('model_path', get_default_model_path())
    sliding_window = request.form.get('sliding_window', 'on') == 'on'
    iou_thres = float(request.form.get('iou_thres', '0.5'))
    min_dist = request.form.get('min_dist', '20')
    min_dist = float(min_dist) if min_dist else 20
    # Visualization options
    viz_conf = float(request.form.get('viz_conf', '0.3'))
    viz_boxes = 'viz_boxes' in request.form or request.method == 'GET'
    viz_contours = ('viz_contours' in request.form) or (request.method == 'GET' and not request.form) or ('run_infer' in request.form) or ('viz_conf' in request.form)
    viz_masks = False
    result_img = None
    geojson_exists = False
    num_detections = 0
    model_paths = get_model_paths()
    geojson_timestamp = None  # Ensure variable is always defined
    
    if request.method == 'POST':
        # 0. Handle file deletion
        if 'delete_file' in request.form and request.form.get('filename'):
            filename = request.form.get('filename')
            if filename and allowed_file(filename):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                if os.path.exists(file_path):
                    os.remove(file_path)
                # Remove associated output files if exist
                base, ext = os.path.splitext(filename)
                geojson_file = os.path.join(app.config['OUTPUT_FOLDER'], f'{base}.geojson')
                vis_file = os.path.join(app.config['OUTPUT_FOLDER'], f'{base}_vis.png')
                for f in [geojson_file, vis_file]:
                    if os.path.exists(f):
                        os.remove(f)
                if session.get('uploaded_img') == filename:
                    session.pop('uploaded_img')
                uploaded_img = None
                uploaded_files = get_uploaded_files()
                # Early return to refresh page after deletion
                return render_template(
                    'index.html',
                    uploaded_img=uploaded_img,
                    result_img=None,
                    conf=viz_conf,
                    viz_conf=viz_conf,
                    viz_boxes=viz_boxes,
                    viz_contours=viz_contours,
                    viz_masks=viz_masks,
                    uploaded_files=uploaded_files,
                    model_path=model_path,
                    sliding_window=sliding_window,
                    iou_thres=iou_thres,
                    min_dist=min_dist,
                    geojson_exists=False,
                    num_detections=0,
                    model_paths=model_paths
                )
        # 1. Run inference if requested
        if 'run_infer' in request.form and request.form.get('filename'):
            filename = request.form.get('filename')
            uploaded_img = filename
            session['uploaded_img'] = filename
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            output_path = app.config['OUTPUT_FOLDER']
            base, ext = os.path.splitext(filename)
            geojson_file = os.path.join(output_path, f'{base}.geojson')
            # Always run inference with conf=0.1
            logging.info(f"Running inference with model: {model_path}")
            cmd = [
                PYTHON_EXECUTABLE,
                'infer_yolo.py',
                '--model', model_path,
                '--input-path', input_path,
                '--output-path', output_path,
                '--conf', '0.1',
                '--iou-thres', str(iou_thres),
                '--min-dist', str(min_dist),
                '--save-geojson', '--no-display'
            ]
            if sliding_window:
                cmd.append('--sliding-window')
            subprocess.run(cmd, cwd=os.path.dirname(__file__))
        # 2. Handle file upload
        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_path)
            session['uploaded_img'] = filename
            uploaded_img = filename
            uploaded_files = get_uploaded_files()  # Refresh the list after upload
        # 3. Handle file selection from dropdown
        if request.form.get('filename') and not 'run_infer' in request.form:
            filename = request.form.get('filename')
            if filename:
                session['uploaded_img'] = filename
                uploaded_img = filename
    # Visualization section
    if uploaded_img:
        filename = uploaded_img
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_path = app.config['OUTPUT_FOLDER']
        base, ext = os.path.splitext(filename)
        geojson_file = os.path.join(output_path, f'{base}.geojson')
        vis_file = f'{base}_vis.png'
        vis_path = os.path.join(output_path, vis_file)
        geojson_exists = os.path.exists(geojson_file)
        geojson_timestamp = None
        if geojson_exists:
            ts = os.path.getmtime(geojson_file)
            geojson_timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            num_detections = plot_detections(
                input_path, geojson_file, vis_path, viz_conf,
                show_boxes=viz_boxes, show_contours=viz_contours, show_masks=viz_masks
            )
            result_img = vis_file
    return render_template(
        'index.html',
        uploaded_img=uploaded_img,
        result_img=result_img,
        conf=viz_conf,
        viz_conf=viz_conf,
        viz_boxes=viz_boxes,
        viz_contours=viz_contours,
        viz_masks=viz_masks,
        uploaded_files=uploaded_files,
        model_path=model_path,
        sliding_window=sliding_window,
        iou_thres=iou_thres,
        min_dist=min_dist,
        geojson_exists=geojson_exists,
        geojson_timestamp=geojson_timestamp,
        num_detections=num_detections,
        model_paths=model_paths
    )

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/viz_update', methods=['POST'])
def viz_update():
    uploaded_img = request.form.get('filename')
    model_path = request.form.get('model_path', get_default_model_path())
    sliding_window = request.form.get('sliding_window', 'on') == 'on'
    iou_thres = float(request.form.get('iou_thres', '0.5'))
    min_dist = request.form.get('min_dist', '20')
    min_dist = float(min_dist) if min_dist else 20
    viz_conf = float(request.form.get('viz_conf', '0.3'))
    viz_boxes = 'viz_boxes' in request.form
    viz_contours = 'viz_contours' in request.form
    viz_masks = False
    result_img = None
    num_detections = 0

    if uploaded_img:
        filename = uploaded_img
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'])
        base, ext = os.path.splitext(filename)
        geojson_file = os.path.join(output_path, f'{base}.geojson')
        vis_file = f'{base}_vis.png'
        vis_path = os.path.join(output_path, vis_file)
        if os.path.exists(geojson_file):
            num_detections = plot_detections(
                input_path, geojson_file, vis_path, viz_conf,
                show_boxes=viz_boxes, show_contours=viz_contours, show_masks=viz_masks
            )
            result_img = vis_file

    return jsonify({
        'result_img_url': url_for('output_file', filename=result_img) if result_img else None,
        'num_detections': num_detections
    })

@app.route('/visualize')
def visualize():
    filename = request.args.get('filename')
    if not filename or not allowed_file(filename):
        return "No valid file selected for visualization.", 400
    base, ext = os.path.splitext(filename)
    vis_file = f'{base}_vis.png'
    geojson_file = os.path.join(app.config['OUTPUT_FOLDER'], f'{base}.geojson')
    vis_path = os.path.join(app.config['OUTPUT_FOLDER'], vis_file)
    if not os.path.exists(vis_path):
        return "No visualization available. Please run inference first.", 404
    viz_conf = float(request.args.get('viz_conf', 0.3))
    viz_boxes = request.args.get('viz_boxes', '1') == '1'
    viz_contours = request.args.get('viz_contours', '1') == '1'
    viz_masks = False
    return render_template(
        'visualize.html',
        filename=filename,
        vis_file=vis_file,
        viz_conf=viz_conf,
        viz_boxes=viz_boxes,
        viz_contours=viz_contours,
        num_detections=plot_detections(
            os.path.join(app.config['UPLOAD_FOLDER'], filename),
            geojson_file,
            vis_path,
            viz_conf,
            show_boxes=viz_boxes,
            show_contours=viz_contours,
            show_masks=viz_masks
        ),
        ts=int(os.path.getmtime(vis_path))
    )

def plot_detections(image_path, geojson_path, out_path, conf_threshold=0.3, show_boxes=True, show_contours=True, show_masks=True):
    img = cv2.imread(image_path)
    if img is None:
        return 0
    with open(geojson_path) as f:
        data = json.load(f)
    detections_found = 0
    for feat in data.get('features', []):
        props = feat.get('properties', {})
        if props.get('confidence', 1.0) < conf_threshold:
            continue
        detections_found += 1
        coords = feat['geometry'].get('coordinates')
        if show_contours and coords and coords[0]:
            pts = np.array(coords[0], dtype=np.int32)
            cv2.polylines(img, [pts], isClosed=True, color=(0,0,255), thickness=2)
        bbox = props.get('bbox')
        if show_boxes and bbox:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        # Masks are not directly supported in geojson, but you could add mask visualization here if needed
    cv2.imwrite(out_path, img)
    return detections_found

#if __name__ == '__main__':
#    app.run(host='0.0.0.0', debug=True)
