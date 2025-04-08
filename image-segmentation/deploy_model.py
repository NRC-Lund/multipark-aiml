from roboflow import Roboflow
import os
from dotenv import load_dotenv

load_dotenv()
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace().project("th-stained-dopamine-neurons-jsfjg")
version = project.version(3)
version.deploy(model_type="yolov11-seg",
               model_path="./runs/segment/train",
               filename="weights/best.pt")