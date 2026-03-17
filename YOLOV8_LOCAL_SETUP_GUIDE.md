# YOLOv8 Local Setup & RC Car Detection Guide

**Date:** March 11, 2026  
**Environment:** Local Development (Mac/Linux/Windows) → Jetson Nano Deployment

---

## Problem Statement

You have local video files of RC cars and want to:
1. Run YOLOv8 inference **locally** to add bounding boxes
2. Test the pipeline before deploying to Jetson Nano
3. Avoid CUDA compatibility issues now (and later on Jetson)

---

## Candidate Solutions: Comprehensive Comparison

### Option 1: YOLOv8 with GPU (CUDA + PyTorch)

**Pros:**
- Fast inference (best performance)
- Full TensorRT export pipeline works
- Can test on your hardware then export for Jetson

**Cons:**
- Requires matching CUDA/cuDNN versions
- GPU-dependent (not portable across systems)
- Setup complexity if CUDA not already installed
- **Jetson Nano CUDA compatibility risk** — Nano often has older CUDA (10.2–11.4), you may be running 12.x locally

**Jetson Nano Compatibility:**
- ❌ **HIGH RISK** — Your local CUDA 12.x models may not run on Nano's CUDA 10.2–11.4

---

### Option 2: YOLOv8 with CPU-Only (PyTorch CPU)

**Pros:**
- Zero CUDA dependency
- Works on any machine (Mac included)
- Safe: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`
- Can develop locally, then test model on Jetson Nano (also CPU-capable)

**Cons:**
- ~10x slower than GPU (inference ~0.5–1 FPS per image on modern CPU)
- Fine for offline video processing (not realtime)
- Not suitable for drone (~15–30 FPS needed)

**Jetson Nano Compatibility:**
- ✅ **PERFECT** — Any CPU model serialization works identically on Nano

---

### Option 3: ONNX Export (Hardware-Agnostic)

**Pros:**
- Export once, run anywhere (CPU/GPU/TPU)
- Works on Jetson Nano without CUDA issues
- Can use CPU inference locally, then optimize with TensorRT on Nano
- Lightweight deployment format

**Cons:**
- Requires ONNX runtime installation
- Extra export step (PyTorch → ONNX)
- Still need GPU for training/fine-tuning locally

**Jetson Nano Compatibility:**
- ✅ **EXCELLENT** — ONNX models run on CPU or TensorRT backend on Nano

---

### Option 4: Use Pre-Optimized Jetson-Nano ONNX Model

**Pros:**
- Model already optimized for Nano
- Plug-and-play deployment
- No training needed (unless fine-tuning)

**Cons:**
- Limited model variety
- May not match exact needs

**Jetson Nano Compatibility:**
- ✅ **BEST** — Built for Nano

---

### Option 5: TensorFlow Lite (Mobile/Edge)

**Pros:**
- Extremely lightweight
- Mobile-optimized
- Works on any embedded hardware

**Cons:**
- Requires YOLOv8 → TFLite conversion (lossy, framework switching)
- Fewer tools/community support vs. PyTorch

**Jetson Nano Compatibility:**
- ⚠️ **OKAY** — Works but not officially supported by Ultralytics

---

## Recommended Path: Option 2 + Option 3

### Phase 1: Local Development (NOW)
- **Use YOLOv8 with CPU-only inference**
- Process your local RC car videos offline
- Add bounding boxes to test detection quality
- No CUDA complications, works on any machine

### Phase 2: Before Jetson Deployment
- **Export model to ONNX**
- Transfer ONNX model to Jetson Nano
- Test inference on Jetson CPU first (verify correctness)
- Then optimize with TensorRT (FP16) for realtime performance

### Why This Works
- **Phase 1:** Fast iteration, zero compatibility issues
- **Phase 2:** Jetson Nano guaranteed to work (ONNX is universal)
- **Phase 2b (optional):** TensorRT optimization for realtime (FP16 → 15–30 FPS)

---

## Step-by-Step Implementation

### Step 1: Environment Setup (Local, CPU-Only)

#### On macOS:
```bash
# Create virtual environment
python3 -m venv yolov8_env
source yolov8_env/bin/activate

# Install PyTorch (CPU-only)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install YOLOv8
pip install ultralytics opencv-python
```

#### On Linux:
```bash
python3 -m venv yolov8_env
source yolov8_env/bin/activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics opencv-python
```

#### On Windows (PowerShell):
```powershell
python -m venv yolov8_env
.\yolov8_env\Scripts\Activate.ps1

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics opencv-python
```

---

### Step 2: Inference on Local Video

Create `detect_rc_car.py`:

```python
#!/usr/bin/env python3
"""
YOLOv8 detection on local video files.
Outputs video with bounding boxes.
"""

from ultralytics import YOLO
import cv2
import os

def detect_on_video(video_path, output_path, model_name='yolov8n.pt'):
    """
    Run YOLOv8 inference on video and save annotated output.
    
    Args:
        video_path: Path to input video file
        output_path: Path to save annotated video
        model_name: YOLOv8 model ('yolov8n', 'yolov8s', etc.)
    """
    
    # Load pretrained model (automatically downloaded on first run)
    # Force CPU device to avoid CUDA issues
    print(f"Loading {model_name} (CPU mode)...")
    model = YOLO(model_name)
    model.to('cpu')  # Explicitly use CPU
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height} @ {fps} FPS, {total_frames} frames")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference (CPU, so will be slower)
        results = model(frame, conf=0.5, device='cpu')
        
        # Draw bounding boxes and labels
        annotated_frame = results[0].plot()
        
        out.write(annotated_frame)
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count}/{total_frames} frames ({100*frame_count/total_frames:.1f}%)")
    
    cap.release()
    out.release()
    print(f"✓ Saved annotated video to: {output_path}")

if __name__ == "__main__":
    # Modify these paths to your local videos
    video_input = "rc_car_video.mp4"
    video_output = "rc_car_video_detected.mp4"
    
    if not os.path.exists(video_input):
        print(f"Error: {video_input} not found")
        print("Place your RC car video in the current directory")
        exit(1)
    
    detect_on_video(video_input, video_output, model_name='yolov8n.pt')
```

**Run it:**
```bash
python detect_rc_car.py
```

---

### Step 3: Evaluate Detection Quality

After generating the annotated video:

1. **Play the output video** and visually inspect:
   - Are RC cars detected consistently?
   - Are false positives acceptable?
   - Is detection confidence high (0.8+)?

2. **Extract metrics** (optional):

```python
#!/usr/bin/env python3
"""
Analyze detection quality on video.
"""

from ultralytics import YOLO
import cv2

def analyze_detections(video_path, model_name='yolov8n.pt'):
    model = YOLO(model_name)
    model.to('cpu')
    
    cap = cv2.VideoCapture(video_path)
    
    detections_per_frame = []
    confidences = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, conf=0.5, device='cpu')
        
        # Filter for 'car' class (COCO class ID 2)
        car_detections = [box for box in results[0].boxes 
                          if int(box.cls) == 2]
        
        detections_per_frame.append(len(car_detections))
        confidences.extend([float(box.conf) for box in car_detections])
    
    cap.release()
    
    print(f"Total frames: {len(detections_per_frame)}")
    print(f"Frames with detections: {sum(1 for x in detections_per_frame if x > 0)}")
    print(f"Avg detections/frame: {sum(detections_per_frame) / len(detections_per_frame):.2f}")
    print(f"Avg confidence: {sum(confidences) / len(confidences):.3f}")
    print(f"Min confidence: {min(confidences):.3f}, Max: {max(confidences):.3f}")

if __name__ == "__main__":
    analyze_detections("rc_car_video.mp4")
```

---

### Step 4: Export to ONNX (for Jetson Nano)

Once satisfied with detection quality:

```python
#!/usr/bin/env python3
"""
Export YOLOv8 model to ONNX format for portable deployment.
"""

from ultralytics import YOLO

def export_to_onnx(model_name='yolov8n.pt', output_dir='./models'):
    """Export YOLOv8 to ONNX format."""
    
    model = YOLO(model_name)
    
    # Export to ONNX
    print(f"Exporting {model_name} to ONNX...")
    export_path = model.export(format='onnx', imgsz=320, device='cpu')
    
    print(f"✓ ONNX model saved to: {export_path}")
    return export_path

if __name__ == "__main__":
    export_to_onnx('yolov8n.pt')
```

**Run it:**
```bash
python export_to_onnx.py
```

This creates `yolov8n.onnx` — a universal model file that runs identically on Jetson Nano.

---

### Step 5: (Optional) Fine-Tune on Your RC Car Dataset

If pretrained YOLOv8n detection is insufficient (high false negatives on RC cars):

#### 5a. Prepare Dataset

```
dataset/
  ├── images/
  │   ├── train/  (300 images)
  │   ├── val/    (50 images)
  │   └── test/   (50 images)
  └── labels/
      ├── train/  (YOLO format .txt)
      ├── val/
      └── test/
```

Each `.txt` file format (YOLO):
```
<class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
0 0.5 0.6 0.2 0.15
```

Use **Roboflow** to auto-generate augmentations and export in YOLO format.

#### 5b. Fine-Tune

```python
#!/usr/bin/env python3
"""
Fine-tune YOLOv8 on custom RC car dataset.
"""

from ultralytics import YOLO

def finetune_yolov8(data_yaml_path, epochs=50, device='cpu'):
    """
    Fine-tune YOLOv8 on custom dataset.
    
    Args:
        data_yaml_path: Path to data.yaml (from Roboflow export)
        epochs: Number of training epochs
        device: 'cpu' or 'cuda:0'
    """
    
    model = YOLO('yolov8n.pt')
    
    print(f"Starting fine-tuning on CPU (slower, but portable)...")
    print("(Consider using GPU if available: device='cuda:0')")
    
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=320,
        device=device,
        patience=10,  # Early stopping
        save=True
    )
    
    print(f"✓ Fine-tuned model saved to: runs/detect/train/weights/best.pt")
    return results

if __name__ == "__main__":
    # Roboflow will provide this after export
    finetune_yolov8('data.yaml', epochs=50, device='cpu')
```

---

## Jetson Nano Deployment (Phase 2)

Once you have `yolov8n.onnx` or fine-tuned `best.pt`:

### On Jetson Nano:

```bash
# SSH into Nano
ssh nano@jetson-nano.local

# Install dependencies
pip install ultralytics onnxruntime

# Transfer model file
scp yolov8n.onnx nano@jetson-nano.local:/home/nano/models/
```

### Inference on Nano (CPU first):

```python
from ultralytics import YOLO

model = YOLO('yolov8n.onnx')  # Load ONNX (works on CPU or GPU)
results = model('video.mp4', device='cpu', conf=0.5)
```

### Then Optimize with TensorRT (for realtime ~20 FPS):

```python
from ultralytics import YOLO

model = YOLO('best.pt')  # Load original
export_path = model.export(format='engine', half=True, device=0)  # FP16 TensorRT
```

---

## Timeline Summary

| Phase | Task | Time | Hardware |
|-------|------|------|----------|
| **1a** | Setup (Step 1) | 10 min | Local CPU |
| **1b** | Test inference on video (Step 2) | 20–30 min | Local CPU |
| **1c** | Evaluate quality (Step 3) | 10 min | Local CPU |
| **1d** | Export to ONNX (Step 4) | 5 min | Local CPU |
| **1e** | (Optional) Fine-tune (Step 5) | 2–4 hrs | Local CPU or GPU |
| **2a** | Deploy to Jetson Nano (ONNX) | 10 min | Jetson Nano |
| **2b** | Test on Nano (CPU) | 5 min | Jetson Nano |
| **2c** | Optimize with TensorRT | 15 min | Jetson Nano |
| **3** | Integrate with drone pipeline | TBD | Jetson Nano |

---

## Expected Performance

| Model | Device | Input Resolution | FPS | Latency |
|-------|--------|-------------------|-----|---------|
| YOLOv8n (CPU) | Local Mac/Linux | 320×320 | 0.5–1 | 1–2 sec |
| YOLOv8n (ONNX CPU) | Jetson Nano | 320×320 | 1–2 | 0.5–1 sec |
| YOLOv8n (TensorRT FP16) | Jetson Nano | 320×320 | 15–30 | 30–70 ms |

**Key:** Phase 1 is slow (offline), but Phase 2c gives realtime performance.

---

## Troubleshooting

### CUDA Version Mismatch
- ✅ **Solution:** Use CPU mode (`device='cpu'`) in Phases 1–2a
- ONNX format avoids CUDA dependency entirely

### Model Not Downloaded
```bash
# If behind firewall, manually download from:
# https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# Then load locally:
model = YOLO('path/to/yolov8n.pt')
```

### Video Writer Codec Issues (Windows)
```python
# Use different codec on Windows
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Instead of 'mp4v'
```

### Jetson Nano Out-of-Memory
```python
# Use smaller model if memory limited
model = YOLO('yolov8n.pt')  # nano (5M params)
# Not: yolov8m.pt (25M) or yolov8l.pt (44M)
```

---

## Next Steps

1. **Immediately (Phase 1):**
   - Run Step 1 (environment setup)
   - Process your RC car video with `detect_rc_car.py`
   - Visually inspect the output

2. **Before Jetson Test:**
   - Export to ONNX (Step 4)
   - Optionally fine-tune if detection poor (Step 5)

3. **On Jetson Nano:**
   - Copy ONNX model
   - Test inference (CPU)
   - Optimize with TensorRT

---

## Files to Create Locally

```
ROB498-roborangers/
├── yolo_scripts/
│   ├── detect_rc_car.py           # Main inference script
│   ├── analyze_detections.py      # Quality evaluation
│   ├── export_to_onnx.py          # Export for Nano
│   └── finetune_yolov8.py         # Optional fine-tuning
├── videos/
│   ├── rc_car_video.mp4           # Your input video
│   └── rc_car_video_detected.mp4  # Output with bboxes
└── models/
    ├── yolov8n.onnx               # Portable model
    └── best.pt                    # Fine-tuned (optional)
```

---

**Ready to start Phase 1?** Begin with Step 1 and report back on detection quality.
