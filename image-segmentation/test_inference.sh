source /srv/data/Resources/Python/anaconda3/bin/activate
conda activate multipark-web 
python ./infer_yolo.py \
 --model './models/th-stained-dopamine-neurons-v3-medium2.pt' \
 --input-path './uploads/442_2.tif' \
 --output-path './test_output' \
 --conf 0.1 \
 --iou-thres 0.5 \
 --min-dist 20 \
 --save-geojson \
 --no-display \
 --sliding-window
