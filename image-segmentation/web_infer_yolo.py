# Minimal Flask web app for infer_yolo
import os
import sys
from flask import Flask, request, render_template_string, send_from_directory, redirect, url_for, session, jsonify
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
  .section { margin-bottom: 32px; }
  .section-title { font-size: 1.2em; font-weight: bold; margin-bottom: 12px; }
  .row { display: flex; flex-wrap: wrap; gap: 24px; align-items: flex-end; }
  .col { flex: 1 1 0; min-width: 200px; }
  img { width: 100%; max-width: 900px; border-radius: 8px; box-shadow: 0 1px 4px #0002; }
  form { margin-bottom: 0; }
  label { font-weight: 500; }
  input[type=range] { width: 80%; }
  .slider-label { display: flex; align-items: center; gap: 12px; margin: 16px 0; }
  fieldset:disabled { opacity: 0.5; }
  .spinner {
    display: none;
    margin: 0 auto;
    border: 6px solid #f3f3f3;
    border-top: 6px solid #007bff;
    border-radius: 50%;
    width: 36px;
    height: 36px;
    animation: spin 1s linear infinite;
  }
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
  .inference-feedback {
    text-align: center;
    margin-top: 12px;
    font-size: 1.1em;
    color: #007bff;
  }
</style>
<div class="container">
  <!-- File selection/upload section -->
  <div class="section">
    <div class="section-title">1. Image files</div>
    <form method=post enctype=multipart/form-data style="margin-bottom:12px;" id="uploadForm">
      <label for="fileInput">Select a file for upload:</label>
      <input type=file name=file accept="image/*" id="fileInput">
      <input type=submit value="Upload New Image" id="uploadBtn" style="display:none;">
    </form>
    <script>
      document.getElementById('fileInput').addEventListener('change', function() {
        if (this.value) {
          document.getElementById('uploadForm').submit();
        }
      });
    </script>
    <form method=post id="fileSelectForm" style="display:inline;">
      <label for="file_select">Or select an already uploaded file:</label>
      <select name="filename" id="file_select" onchange="this.form.submit()">
        <option value="">-- Select file --</option>
        {% for fname in uploaded_files %}
          <option value="{{ fname }}" {% if fname == uploaded_img %}selected{% endif %}>{{ fname }}</option>
        {% endfor %}
      </select>
      <button type="submit" name="delete_file" value="1" onclick="return confirm('Are you sure you want to delete this file?');" {% if not uploaded_img %}disabled{% endif %}>Delete selected file</button>
    </form>
  </div>

  <!-- Inference options section -->
  <div class="section">
    <div class="section-title">2. Inference Options</div>
    <form method=post id="inferForm">
      <input type="hidden" name="filename" value="{{ uploaded_img }}">
      <div class="row">
        <div class="col">
          <label>Model:</label>
          <select name="model_path" style="width:100%">
            {% for mpath in model_paths %}
              <option value="{{ mpath }}" {% if mpath == model_path %}selected{% endif %}>{{ mpath.split('/')[-1] }}</option>
            {% endfor %}
          </select>
        </div>
        <div class="col">
          <label><input type="checkbox" name="sliding_window" {% if sliding_window %}checked{% endif %}> Sliding Window</label>
        </div>
        <div class="col">
          <label>IOU Threshold:</label><br>
          <input type="number" name="iou_thres" min="0" max="1" step="0.01" value="{{ iou_thres }}">
        </div>
        <div class="col">
          <label>Min Distance:</label><br>
          <input type="number" name="min_dist" min="0" step="1" value="{{ min_dist }}">
        </div>
        <div class="col">
          <input type="submit" name="run_infer" value="Run Inference on selected image">
        </div>
      </div>
      <div class="inference-feedback" id="inferenceFeedback">
        <div class="spinner" id="spinner"></div>
        <span id="inferenceMsg"></span>
      </div>
    </form>
  </div>

  <!-- Visualization section -->
  <div class="section">
    <div class="section-title">3. Visualization</div>
    <fieldset {% if not geojson_exists %}disabled{% endif %}>
      <form method=post id="vizform" onsubmit="return false;">
        <input type="hidden" name="filename" value="{{ uploaded_img }}">
        <input type="hidden" name="model_path" value="{{ model_path }}">
        <input type="hidden" name="sliding_window" value="{{ 'on' if sliding_window else '' }}">
        <input type="hidden" name="iou_thres" value="{{ iou_thres }}">
        <input type="hidden" name="min_dist" value="{{ min_dist }}">
        <div class="row">
          <div class="col">
            <div class="slider-label">
              <label>Confidence Threshold:</label>
              <input type="range" name="viz_conf" min="0" max="1" step="0.01" value="{{ viz_conf }}" id="viz_conf_slider">
              <span id="viz_conf_val">{{ viz_conf }}</span>
            </div>
          </div>
          <div class="col">
            <label><input type="checkbox" name="viz_boxes" {% if viz_boxes %}checked{% endif %} id="viz_boxes_cb"> Show Boxes</label>
          </div>
          <div class="col">
            <label><input type="checkbox" name="viz_contours" {% if viz_contours %}checked{% endif %} id="viz_contours_cb"> Show Contours</label>
          </div>
        </div>
      </form>
      <div id="viz_result">
      {% if result_img %}
        <h2>Detections:</h2>
        <img src="{{ url_for('output_file', filename=result_img) }}" id="viz_img">
        <div style="font-size:1em;margin-top:12px;">
          {{ num_detections }} detection{{ 's' if num_detections != 1 else '' }} found at this confidence threshold.
        </div>
      {% endif %}
      </div>
      {% if not geojson_exists %}
        <div style="color:#c00;font-size:1em;margin-top:12px;">Run inference to enable visualization.</div>
      {% endif %}
      <script>
        function sendVizAjax() {
          const form = document.getElementById('vizform');
          const formData = new FormData(form);
          fetch('/viz_update', {
            method: 'POST',
            body: formData
          })
          .then(response => response.json())
          .then(data => {
            let html = '';
            if (data.result_img_url) {
              html += '<h2>Detections:</h2>';
              html += `<img src="${data.result_img_url}?t=${Date.now()}" id="viz_img">`;
              html += `<div style=\"font-size:1em;margin-top:12px;\">${data.num_detections} detection${data.num_detections==1?'':'s'} found at this confidence threshold.</div>`;
            }
            document.getElementById('viz_result').innerHTML = html;
          });
        }
        document.getElementById('viz_conf_slider').oninput = function(e) {
          document.getElementById('viz_conf_val').innerText = this.value;
        };
        document.getElementById('viz_conf_slider').addEventListener('change', function(e) {
          e.preventDefault();
          e.stopPropagation();
          sendVizAjax();
          return false;
        });
        document.getElementById('viz_boxes_cb').addEventListener('change', function(e) {
          e.preventDefault();
          e.stopPropagation();
          sendVizAjax();
          return false;
        });
        document.getElementById('viz_contours_cb').addEventListener('change', function(e) {
          e.preventDefault();
          e.stopPropagation();
          sendVizAjax();
          return false;
        });
      </script>
    </fieldset>
  </div>
</div>
<script>
  document.getElementById('inferForm').addEventListener('submit', function() {
    document.getElementById('spinner').style.display = 'inline-block';
    document.getElementById('inferenceMsg').textContent = 'Running inference, please wait...';
  });
  window.onload = function() {
    document.getElementById('spinner').style.display = 'none';
    document.getElementById('inferenceMsg').textContent = '';
  };
</script>
'''

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_default_model_path():
    return '/Users/med-pha/Documents/Science/Projects/ImageSegmentation/models/th-stained-dopamine-neurons-v3-medium.pt'

def get_uploaded_files():
    files = []
    for fname in os.listdir(UPLOAD_FOLDER):
        if allowed_file(fname):
            files.append(fname)
    return sorted(files)

def get_model_paths():
    model_dir = '/Users/med-pha/Documents/Science/Projects/ImageSegmentation/models'
    return [os.path.join(model_dir, fname) for fname in os.listdir(model_dir) if fname.endswith('.pt')]

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
    viz_masks = 'viz_masks' in request.form or request.method == 'GET'
    result_img = None
    geojson_exists = False
    num_detections = 0
    model_paths = get_model_paths()
    
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
                return render_template_string(
                    HTML,
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
            cmd = [
                sys.executable, 'infer_yolo.py',
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
        if geojson_exists:
            num_detections = plot_detections(
                input_path, geojson_file, vis_path, viz_conf,
                show_boxes=viz_boxes, show_contours=viz_contours, show_masks=viz_masks
            )
            result_img = vis_file
    return render_template_string(
        HTML,
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
    viz_masks = 'viz_masks' in request.form
    result_img = None
    num_detections = 0

    if uploaded_img:
        filename = uploaded_img
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_path = app.config['OUTPUT_FOLDER']
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

if __name__ == '__main__':
    app.run(debug=True)
