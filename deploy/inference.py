"""
Run HybridEnhancementNet inference on rover camera feed or image files.

Usage:
    # Single image
    python deploy/inference.py --model models/exported/model.onnx --source image.jpg

    # Folder of images
    python deploy/inference.py --model models/exported/model.onnx --source ./frames/

    # Live camera (rover)
    python deploy/inference.py --model models/exported/model.onnx --source camera
"""

import argparse
import os
import time
import numpy as np
from PIL import Image


def load_model(model_path):
    import onnxruntime as ort
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    print(f"Loaded ONNX model: {model_path}")
    return session


def preprocess(image_path_or_array, size=(256, 256)):
    """Load and preprocess image to model input format."""
    if isinstance(image_path_or_array, str):
        img = Image.open(image_path_or_array).convert('RGB')
    else:
        img = Image.fromarray(image_path_or_array)

    img   = img.resize(size)
    arr   = np.array(img).astype(np.float32) / 255.0
    arr   = arr.transpose(2, 0, 1)   # HWC -> CHW
    arr   = np.expand_dims(arr, 0)   # add batch dim
    return arr


def postprocess(output_array):
    """Convert model output back to uint8 image."""
    out = output_array[0]           # remove batch dim
    out = out.transpose(1, 2, 0)   # CHW -> HWC
    out = (out + 1.0) / 2.0        # tanh [-1,1] -> [0,1]
    out = np.clip(out * 255, 0, 255).astype(np.uint8)
    return out


def enhance_image(session, img_input):
    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result      = session.run([output_name], {input_name: img_input})
    return result[0]


def run_on_folder(session, folder_path, output_dir, size=(256, 256)):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for fname in files:
        t0  = time.time()
        inp = preprocess(os.path.join(folder_path, fname), size)
        out = enhance_image(session, inp)
        img = postprocess(out)
        Image.fromarray(img).save(os.path.join(output_dir, f'enhanced_{fname}'))
        print(f"{fname} -> {time.time()-t0:.3f}s")


def run_camera(session, size=(256, 256)):
    import cv2
    cap = cv2.VideoCapture(0)
    print("Running on camera. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp       = preprocess(frame_rgb, size)
        out       = enhance_image(session, inp)
        enhanced  = postprocess(out)

        # Resize back to display size
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
        display      = cv2.resize(enhanced_bgr, (frame.shape[1], frame.shape[0]))
        combined     = np.hstack([frame, display])

        cv2.imshow('Original | Enhanced', combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',  required=True, help='Path to ONNX model')
    parser.add_argument('--source', required=True, help='Image file, folder, or "camera"')
    parser.add_argument('--output', default='./enhanced_output')
    parser.add_argument('--size',   nargs=2, type=int, default=[256, 256])
    args = parser.parse_args()

    session = load_model(args.model)
    size    = tuple(args.size)

    if args.source == 'camera':
        run_camera(session, size)
    elif os.path.isdir(args.source):
        run_on_folder(session, args.source, args.output, size)
    else:
        inp = preprocess(args.source, size)
        out = enhance_image(session, inp)
        img = postprocess(out)
        out_path = os.path.join(args.output, f'enhanced_{os.path.basename(args.source)}')
        os.makedirs(args.output, exist_ok=True)
        Image.fromarray(img).save(out_path)
        print(f"Saved enhanced image to {out_path}")
