## Run centroid detector node and yolov8n inference:

```bash
python3 imx219_yolov8n_trt.py --model ~/yolov8n.engine --input-size 320

# Or with the alias name:
python3 imx219_yolov8_trt.py --model ~/yolov8n.engine --input-size 320

python3 imx219_yolov8n_trt.py --model ~/yolov8n.engine --class-names /path/to/classes.txt --target-class-name cars --input-size 320
```
