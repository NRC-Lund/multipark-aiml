# Minimal Flask web app for infer_yolo
import os
from flask import Flask, request, render_template_string, send_from_directory, redirect, url_for, session
import subprocess
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import cv2
import numpy as np
import json

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.secret_key = 'your_secret_key'  # Needed for session

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

HTML = '''
<!doctype html>
<title>YOLO Inference Web</title>
<style>
  body { font-family: sans-serif; margin: 0; background: #f5f5f5; }
  .container { max-width: 900px; margin: 30px auto; background: #fff; border-radius: 8px; box-shadow: 0 2px 8px #0001; padding: 32px; }
  .col { width: 100%; }
  img { width: 100%; max-width: 900px; border-radius: 8px; box-shadow: 0 1px 4px #0002; }
  form { margin-bottom: 24px; }
  label { font-weight: 500; }
  input[type=range] { width: 80%; }
  .slider-label { display: flex; align-items: center; gap: 12px; margin: 16px 0; }
</style>
<div class="container">
  <h1>YOLO Inference Web</h1>
  <form method=post enctype=multipart/form-data>
    <input type=file name=file accept="image/*">
    <input type=submit value="Upload Image">
  </form>
  {% if uploaded_img %}
    <div class="col">
      <form method=post>
        <input type="hidden" name="filename" value="{{ uploaded_img }}">
        <div class="slider-label">
          <label>Confidence Threshold:</label>
          <input type="range" name="conf" min="0" max="1" step="0.01" value="{{ conf }}" oninput="document.getElementById('conf_val').innerText=this.value">
          <span id="conf_val">{{ conf }}</span>
        </div>
        <input type="submit" name="run_infer" value="Run Inference">
      </form>
      {% if result_img %}
        <h2>Detections:</h2>
        <img src="{{ url_for('output_file', filename=result_img) }}">
      {% endif %}
    </div>
  {% endif %}
</div>
'''

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_and_infer():
    uploaded_img = session.get('uploaded_img')
    result_img = None
    conf = float(request.form.get('conf', '0.1'))
    if request.method == 'POST':
        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_path)
            session['uploaded_img'] = filename
            uploaded_img = filename
            result_img = None
        elif 'run_infer' in request.form and uploaded_img:
            filename = uploaded_img
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            output_path = app.config['OUTPUT_FOLDER']
            model_path = '/Users/med-pha/Documents/Science/Projects/ImageSegmentation/models/th-stained-dopamine-neurons-v3-medium.pt'
            base, ext = os.path.splitext(filename)
            geojson_file = os.path.join(output_path, f'{base}.geojson')
            vis_file = f'{base}_vis.png'
            vis_path = os.path.join(output_path, vis_file)
            # Only run inference if geojson does not exist
            if not os.path.exists(geojson_file):
                cmd = [
                    'python', 'infer_yolo.py',
                    '--model', model_path,
                    '--input-path', input_path,
                    '--output-path', output_path,
                    '--conf', str(conf),
                    '--iou-thres', '0.5',
                    '--min-dist', '20',
                    '--save-geojson', '--no-display', '--sliding-window', '--no-colors'
                ]
                subprocess.run(cmd, cwd=os.path.dirname(__file__))
            # Always re-plot using the current confidence
            if os.path.exists(geojson_file):
                plot_detections(input_path, geojson_file, vis_path, conf)
                result_img = vis_file
    return render_template_string(HTML, uploaded_img=uploaded_img, result_img=result_img, conf=conf)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

def plot_detections(image_path, geojson_path, out_path, conf_threshold=0.3):
    img = cv2.imread(image_path)
    if img is None:
        return
    with open(geojson_path) as f:
        data = json.load(f)
    for feat in data.get('features', []):
        props = feat.get('properties', {})
        if props.get('confidence', 1.0) < conf_threshold:
            continue
        coords = feat['geometry'].get('coordinates')
        if coords and coords[0]:
            pts = np.array(coords[0], dtype=np.int32)
            cv2.polylines(img, [pts], isClosed=True, color=(0,0,255), thickness=2)
        bbox = props.get('bbox')
        if bbox:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.imwrite(out_path, img)

if __name__ == '__main__':
    app.run(debug=True)
