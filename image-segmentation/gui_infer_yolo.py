import webview
import json
import os
import sys
import subprocess
from pathlib import Path

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>YOLO Inference GUI</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: 500;
        }
        input[type="text"],
        input[type="number"] {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }
        .file-input {
            display: flex;
            gap: 10px;
        }
        .file-input input[type="text"] {
            flex: 1;
        }
        button {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        button:hover {
            background-color: #0056b3;
        }
        .tabs {
            display: flex;
            margin-bottom: 20px;
            border-bottom: 1px solid #ddd;
        }
        .tab {
            padding: 10px 20px;
            cursor: pointer;
            border: none;
            background: none;
            font-size: 14px;
            color: #666;
        }
        .tab.active {
            color: #007bff;
            border-bottom: 2px solid #007bff;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .checkbox-group {
            margin-bottom: 10px;
        }
        .checkbox-group label {
            display: inline-block;
            margin-left: 5px;
        }
        #status {
            margin-top: 20px;
            padding: 10px;
            border-radius: 4px;
            background-color: #f8f9fa;
        }
        .color-input {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        .color-input input {
            width: 100px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="form-group">
            <label>Model:</label>
            <div class="file-input">
                <input type="text" id="model-path" readonly>
                <button onclick="browseModel()">Browse</button>
            </div>
        </div>

        <div class="form-group">
            <label>Input Path (file or folder):</label>
            <div class="file-input">
                <input type="text" id="input-file-path" readonly>
                <button onclick="browseInputFile()">Browse File</button>
                <button onclick="browseInputFolder()">Browse Folder</button>
            </div>
        </div>

        <div class="form-group">
            <label>Output Path (folder):</label>
            <div class="file-input">
                <input type="text" id="output-path" readonly>
                <button onclick="browseOutput()">Browse</button>
            </div>
        </div>

        <div class="tabs">
            <button class="tab active" onclick="showTab('basic')">Basic</button>
            <button class="tab" onclick="showTab('visualization')">Visualization</button>
            <button class="tab" onclick="showTab('output')">Output</button>
        </div>

        <div id="basic" class="tab-content active">
            <div class="form-group">
                <label>Confidence Threshold:</label>
                <input type="number" id="conf-threshold" value="0.25" step="0.01">
            </div>
            <div class="checkbox-group">
                <input type="checkbox" id="use-sliding-window" checked>
                <label>Use Sliding Window</label>
            </div>
            <div class="form-group">
                <label>Window Size:</label>
                <input type="number" id="window-size" value="640">
            </div>
            <div class="form-group">
                <label>Overlap:</label>
                <input type="number" id="overlap" value="128">
            </div>
        </div>

        <div id="visualization" class="tab-content">
            <div class="checkbox-group">
                <input type="checkbox" id="draw-boxes" checked>
                <label>Draw Boxes</label>
            </div>
            <div class="checkbox-group">
                <input type="checkbox" id="draw-labels" checked>
                <label>Draw Labels</label>
            </div>
            <div class="checkbox-group">
                <input type="checkbox" id="draw-contours" checked>
                <label>Draw Contours</label>
            </div>
            <div class="checkbox-group">
                <input type="checkbox" id="draw-masks" checked>
                <label>Draw Masks</label>
            </div>
            <div class="checkbox-group">
                <input type="checkbox" id="no-colors">
                <label>Convert to Grayscale</label>
            </div>
            <div class="checkbox-group">
                <input type="checkbox" id="no-display" checked>
                <label>Hide Display</label>
            </div>
            <div class="form-group">
                <label>Box Color (B,G,R):</label>
                <input type="text" id="box-color" value="0,255,0">
            </div>
            <div class="form-group">
                <label>Contour Color (B,G,R):</label>
                <input type="text" id="contour-color" value="0,0,255">
            </div>
            <div class="form-group">
                <label>Contour Thickness:</label>
                <input type="number" id="contour-thickness" value="1">
            </div>
            <div class="form-group">
                <label>Mask Alpha:</label>
                <input type="number" id="mask-alpha" value="0.3" step="0.1">
            </div>
        </div>

        <div id="output" class="tab-content">
            <div class="checkbox-group">
                <input type="checkbox" id="save-image" checked>
                <label>Save Visualization Image</label>
            </div>
            <div class="checkbox-group">
                <input type="checkbox" id="save-geojson" checked>
                <label>Save GeoJSON Results</label>
            </div>
        </div>

        <button onclick="runInference()" style="width: 100%; margin-top: 20px;">Run Inference</button>
        <div id="status">Ready</div>
    </div>

    <script>
        function showTab(tabId) {
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            // Show selected tab content
            document.getElementById(tabId).classList.add('active');
            
            // Update tab buttons
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            event.target.classList.add('active');
        }

        function browseModel() {
            pywebview.api.browseModel().then(path => {
                if (path) document.getElementById('model-path').value = path;
            });
        }

        function browseInputFile() {
            pywebview.api.browseInputFile().then(path => {
                if (path) document.getElementById('input-file-path').value = path;
            });
        }

        function browseInputFolder() {
            pywebview.api.browseInputFolder().then(path => {
                if (path) document.getElementById('input-file-path').value = path;
            });
        }

        function browseOutput() {
            pywebview.api.browseOutput().then(path => {
                if (path) document.getElementById('output-path').value = path;
            });
        }

        function runInference() {
            const config = {
                model_path: document.getElementById('model-path').value,
                input_path: document.getElementById('input-file-path').value,
                output_path: document.getElementById('output-path').value,
                conf_threshold: parseFloat(document.getElementById('conf-threshold').value),
                use_sliding_window: document.getElementById('use-sliding-window').checked,
                window_size: parseInt(document.getElementById('window-size').value),
                overlap: parseInt(document.getElementById('overlap').value),
                draw_boxes: document.getElementById('draw-boxes').checked,
                draw_labels: document.getElementById('draw-labels').checked,
                draw_contours: document.getElementById('draw-contours').checked,
                draw_masks: document.getElementById('draw-masks').checked,
                no_colors: document.getElementById('no-colors').checked,
                no_display: document.getElementById('no-display').checked,
                box_color: document.getElementById('box-color').value,
                contour_color: document.getElementById('contour-color').value,
                contour_thickness: parseInt(document.getElementById('contour-thickness').value),
                mask_alpha: parseFloat(document.getElementById('mask-alpha').value),
                save_image: document.getElementById('save-image').checked,
                save_geojson: document.getElementById('save-geojson').checked
            };

            document.getElementById('status').textContent = 'Running inference...';
            pywebview.api.runInference(config).then(result => {
                document.getElementById('status').textContent = result;
            });
        }
    </script>
</body>
</html>
"""

class YOLOInferenceGUI:
    def __init__(self):
        self.window = webview.create_window(
            'YOLO Inference GUI',
            html=HTML,
            js_api=self
        )
    
    def browseModel(self):
        return webview.windows[0].create_file_dialog(
            webview.OPEN_DIALOG,
            directory='./models',
            file_types=('PyTorch model files (*.pt)', 'All files (*.*)')
        )
    
    def browseInputFile(self):
        return webview.windows[0].create_file_dialog(
            webview.OPEN_DIALOG,
            file_types=('Image files (*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff)', 'All files (*.*)')
        )
    
    def browseInputFolder(self):
        return webview.windows[0].create_file_dialog(
            webview.FOLDER_DIALOG
        )
    
    def browseOutput(self):
        return webview.windows[0].create_file_dialog(
            webview.SAVE_DIALOG
        )
    
    def runInference(self, config):
        try:
            # Build command
            script_dir = os.path.dirname(os.path.abspath(__file__))
            infer_yolo_path = os.path.join(script_dir, "infer_yolo.py")
            cmd = [
                sys.executable, infer_yolo_path,
                "--model", config['model_path'],
                "--input-path", config['input_path'],
                "--output-path", config['output_path'],
                "--conf", str(config['conf_threshold'])
            ]
            
            # Add sliding window options if enabled
            if config['use_sliding_window']:
                cmd.extend([
                    "--sliding-window",
                    "--window-size", str(config['window_size']),
                    "--overlap", str(config['overlap'])
                ])
            
            # Add visualization options
            if not config['draw_boxes']:
                cmd.append("--no-boxes")
            if not config['draw_labels']:
                cmd.append("--no-labels")
            if not config['draw_contours']:
                cmd.append("--no-contours")
            if not config['draw_masks']:
                cmd.append("--no-masks")
            if config['no_colors']:
                cmd.append("--no-colors")
            if config['no_display']:
                cmd.append("--no-display")
            
            # Add colors and styles
            cmd.extend([
                "--box-color", config['box_color'],
                "--contour-color", config['contour_color'],
                "--contour-thickness", str(config['contour_thickness']),
                "--mask-alpha", str(config['mask_alpha'])
            ])
            
            # Add output options
            if config['save_image']:
                cmd.append("--save-image")
            if config['save_geojson']:
                cmd.append("--save-geojson")
            
            # Run the command
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Read output in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
            
            # Get return code
            return_code = process.poll()
            
            if return_code == 0:
                return "Inference completed successfully"
            else:
                error_output = process.stderr.read()
                return f"Inference failed with error:\n{error_output}"
        
        except Exception as e:
            return f"An error occurred:\n{str(e)}"

def main():
    gui = YOLOInferenceGUI()
    webview.start(debug=True)

if __name__ == "__main__":
    main()