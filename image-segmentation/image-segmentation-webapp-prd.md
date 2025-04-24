# 📝 Product Requirements Document (PRD)

## 📌 Project Title:
**Image Segmentation Web App using YOLO**

---

## 📣 Purpose
To provide an interactive web-based tool that allows users to upload images and perform object detection and segmentation using a pre-trained YOLO model.

---

## 🎯 Goals & Objectives

1. Enable users to upload images via a web interface.
2. Perform object detection and segmentation on uploaded images using a YOLO-based model.
3. Visualize inference results.
4. Ensure compatibility across development and production environments.
5. Support SSL-secured access via Apache and WSGI integration.

---

## 🛠️ Features

### 1. Image Upload
- Users can upload `.jpg`, `.png`, `.tif` and `.jpeg` images.
- Uploaded images are stored on the server (`uploads/`).

### 2. YOLO Inference
- Backend Python script (`infer_yolo.py`) performs:
  - Image pre-processing
  - Sliding window inference
  - Non-Maximum Suppression (NMS)
  - Polygon-based result aggregation

### 3. Web Interface
- Flask-based frontend (`web_infer_yolo.py`) allows:
  - Upload
  - Triggering inference
  - Displaying output

### 4. Result Handling
- Annotated output images are saved in the `outputs/` folder.
- Results include visual bounding boxes or polygons around detected objects.

### 5. Apache + WSGI Deployment
- Apache virtual host config (`img-seg.conf`) serves:
  - Web app (`webapp.wsgi`)
  - Static files and output folders
- WSGI integration is configured (`02-wsgi.conf`) for Python 3.11 and Anaconda environment.

---

## ⚙️ Architecture

```
User (Web Browser)
      |
      v
+-----------------+
| Flask Frontend  |  <- web_infer_yolo.py
+-----------------+
      |
      v
+-----------------+
| Inference Logic |  <- infer_yolo.py
+-----------------+
      |
      v
[ YOLO Model / Polygon NMS / Sliding Window ]
      |
      v
[ Outputs: Annotated images ]
```

- **Local (Dev)**: Flask runs directly, using `.env` or local paths.
- **Production**: Apache + mod_wsgi serve the app via HTTPS, also using `.env` or local paths.

---

## 📁 File Structure

```
image-segmentation/
├── infer_yolo.py
├── web_infer_yolo.py
├── webapp.wsgi
├── models/
├── outputs/
├── uploads/
├── static/
├── .env (optional for config)
```

---

## 🌐 Deployment Details

- **Server**: Apache with mod_wsgi
- **OS**: Linux (RHEL/CentOS)
- **Python Environment**: Anaconda (Python 3.11)
- **SSL**: Enabled with certificates from `/etc/ssl/`

---

## 🔒 Security Considerations

- Uploaded images are validated before processing.
- Output and model directories are served read-only via Apache.
- WSGI process runs under limited privileges (`user=apache`).

---

## 🔁 Development vs. Deployment

| Feature         | Development (macOS)               | Deployment (Linux)                                  |
|----------------|------------------------------------|-----------------------------------------------------|
| Web server     | Flask (`flask run`)                | Apache + mod_wsgi                                   |
| File paths     | Relative or local absolute         | Apache-mapped absolute paths (from config)          |
| Config         | `.env` with fallback defaults      | Apache environment vars (`SetEnv`) or hardcoded     |

---

## 🧪 Testing & Validation

- Unit test inference logic (`infer_yolo.py`)
- Upload/download flow tested via Flask in dev and Apache in prod
- Test with:
  - Valid images
  - Corrupted files
  - Edge-case polygons (invalid, low-coord, non-closed)

---

## 🔜 Future Improvements

- Prevent interference between multiple users