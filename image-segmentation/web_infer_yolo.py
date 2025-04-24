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
import datetime
from dotenv import load_dotenv
from config import DevelopmentConfig, ProductionConfig

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

app = Flask(__name__)

# Set secret key from environment variable
app.secret_key = os.getenv('SECRET_KEY', 'your_secret_key')  # Fallback for safety

# Choose config based on environment variable
if os.getenv('FLASK_ENV') == 'production':
    app.config.from_object(ProductionConfig)
else:
    app.config.from_object(DevelopmentConfig)

UPLOAD_FOLDER = app.config['UPLOAD_FOLDER']
OUTPUT_FOLDER = app.config['OUTPUT_FOLDER']
MODEL_FOLDER = app.config['MODEL_FOLDER']
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}

print(f"Upload folder: {UPLOAD_FOLDER}")
print(f"Output folder: {OUTPUT_FOLDER}")
print(f"Model folder: {MODEL_FOLDER}")

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
      <label for="fileInput">Upload a file:</label>
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
    {% if uploaded_img %}
      <div style="margin-top:10px; max-width:200px;">
        <img src="{{ url_for('uploaded_file', filename=uploaded_img) }}" alt="Preview" style="width:100%; border-radius:4px; box-shadow:0 1px 4px #0002;">
      </div>
    {% endif %}
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
          <input type="submit" name="run_infer" value="Run inference on selected image">
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
    {% if uploaded_img %}
      {% if geojson_exists %}
        <div style="color: #228B22; margin-bottom: 10px;">
          Inference has been calculated for this image.<br>
          <span style="font-size:0.95em; color:#555;">
            Timestamp: {{ geojson_timestamp }}
          </span>
        </div>
      {% else %}
        <div style="color: #B22222; margin-bottom: 10px;">You have to run inference before visualizing.</div>
      {% endif %}
    {% endif %}
    <form method="get" action="{{ url_for('visualize') }}">
      <input type="hidden" name="filename" value="{{ uploaded_img }}">
      <button type="submit" {% if not geojson_exists %}disabled{% endif %}>Visualize</button>
    </form>
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
            print(model_path)
            cmd = [
                sys.executable,
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
    return render_template_string('''
    <!doctype html>
    <title>Visualization</title>
    <style>
      body { font-family: sans-serif; margin: 0; background: #f5f5f5; }
      .container { max-width: 900px; margin: 40px auto; background: #fff; padding: 32px; border-radius: 8px; box-shadow: 0 2px 8px #0001; }
      .section-title { font-size: 1.2em; font-weight: bold; margin-bottom: 12px; }
      img { width: 100%; max-width: 900px; border-radius: 8px; box-shadow: 0 1px 4px #0002; }
      .slider-label { display: flex; align-items: center; gap: 12px; margin: 16px 0; }
      .checkbox-group { display: flex; gap: 24px; margin: 16px 0; }
      .inference-feedback { text-align: center; margin-top: 12px; font-size: 1.1em; color: #007bff; }
      .back-link { margin-top: 24px; display: block; }
    </style>
    <div class="container">
      <div class="section-title">Visualization for: {{ filename }}</div>
      <form id="vizForm">
        <div class="slider-label">
          <label for="viz_conf_slider">Confidence threshold:</label>
          <input type="range" min="0" max="1" step="0.01" value="{{ viz_conf }}" id="viz_conf_slider" name="viz_conf">
          <span id="viz_conf_val">{{ viz_conf }}</span>
        </div>
        <div class="checkbox-group">
          <label><input type="checkbox" id="viz_boxes_cb" name="viz_boxes" {% if viz_boxes %}checked{% endif %}> Show boxes</label>
          <label><input type="checkbox" id="viz_contours_cb" name="viz_contours" {% if viz_contours %}checked{% endif %}> Show contours</label>
        </div>
        <input type="hidden" name="filename" value="{{ filename }}">
      </form>
      <div id="viz_result">
        <img src="{{ url_for('output_file', filename=vis_file) }}?t={{ ts }}" id="viz_img">
        <div style="font-size:1em;margin-top:12px;">{{ num_detections }} detection{{ '' if num_detections==1 else 's' }} found at this confidence threshold.</div>
      </div>
      <a href="{{ url_for('upload_and_infer') }}" class="back-link">&larr; Back to main page</a>
    </div>
    <script>
      function sendVizAjax() {
        const form = document.getElementById('vizForm');
        const formData = new FormData(form);
        formData.set('filename', '{{ filename }}'); // Ensure filename is always sent
        if (document.getElementById('viz_boxes_cb').checked) {
          formData.set('viz_boxes', 'on');
        } else {
          formData.delete('viz_boxes');
        }
        if (document.getElementById('viz_contours_cb').checked) {
          formData.set('viz_contours', 'on');
        } else {
          formData.delete('viz_contours');
        }
        formData.set('viz_conf', document.getElementById('viz_conf_slider').value);
        fetch('{{ url_for("viz_update") }}', {
          method: 'POST',
          body: formData
        })
        .then(response => response.json())
        .then(data => {
          if (data.result_img_url) {
            const img = new Image();
            img.id = "viz_img";
            img.src = data.result_img_url + "?t=" + Date.now();
            img.style.width = "100%";
            img.style.maxWidth = "900px";
            img.style.borderRadius = "8px";
            img.style.boxShadow = "0 1px 4px #0002";
            img.onload = function() {
              const vizResult = document.getElementById('viz_result');
              // Replace only the image, keep the detection count
              const countDiv = vizResult.querySelector('div') || document.createElement('div');
              countDiv.style.fontSize = "1em";
              countDiv.style.marginTop = "12px";
              countDiv.innerHTML = `${data.num_detections} detection${data.num_detections==1?'':'s'} found at this confidence threshold.`;
              vizResult.innerHTML = '';
              vizResult.appendChild(img);
              vizResult.appendChild(countDiv);
            };
          }
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
    ''',
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
