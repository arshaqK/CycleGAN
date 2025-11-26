# app.py
import os
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from models import ResnetGenerator
import utils
import cv2
import numpy as np

app = Flask(__name__)
CHECKPOINT_DIR = '/content/drive/MyDrive/cycle_checkpoints'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load latest checkpoint
def load_models(checkpoint_dir=CHECKPOINT_DIR):
    ck = utils.find_latest_checkpoint(checkpoint_dir)
    if ck is None:
        raise FileNotFoundError("No checkpoint found. Train the model first.")
    data = torch.load(ck, map_location=device)
    G_A2B = ResnetGenerator(3,3).to(device)
    G_B2A = ResnetGenerator(3,3).to(device)
    G_A2B.load_state_dict(data['netG_A2B'])
    G_B2A.load_state_dict(data['netG_B2A'])
    G_A2B.eval(); G_B2A.eval()
    return G_A2B, G_B2A

G_A2B, G_B2A = load_models()

# transforms
preprocess = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

postprocess = lambda t: ((t + 1) / 2).clamp(0,1)

# heuristic to detect sketch vs real photo
def is_sketch(pil_img):
    # convert to np RGB
    img = np.array(pil_img.convert('RGB'))
    # saturation heuristic (HSV): sketches often have low saturation
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    sat_mean = hsv[...,1].mean()
    # edge density (Canny)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.mean() / 255.0
    # combined heuristic thresholds (tuned)
    if sat_mean < 30 or edge_density > 0.06:
        return True
    return False

def run_inference(pil_img, to_sketch=False):
    img = pil_img.convert('RGB')
    t = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        if to_sketch:
            out = G_B2A(t)
        else:
            out = G_A2B(t)
    out_img = postprocess(out.squeeze(0)).cpu()
    # convert to PIL
    out_img_pil = transforms.ToPILImage()(out_img)
    return out_img_pil

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/convert', methods=['POST'])
def convert():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    pil_img = Image.open(file.stream)
    sketch_detected = is_sketch(pil_img)
    if sketch_detected:
        output = run_inference(pil_img, to_sketch=False)  # sketch -> photo
        domain = 'sketch'
        converted_to = 'photo'
    else:
        output = run_inference(pil_img, to_sketch=True)   # photo -> sketch
        domain = 'photo'
        converted_to = 'sketch'
    # send image bytes back
    img_io = io.BytesIO()
    output.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

if __name__ == "__main__":
    # run on 0.0.0.0 for Colab tunneling
    app.run(host='127.0.0.1', port=5000, debug=True)
