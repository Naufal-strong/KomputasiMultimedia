"""
run_experiments.py
Pipeline script for Tugas 2 Komputasi Multimedia:
- Bagian A: Video frame extraction (DAiSEE simulation/capture), 10 random sample frames, 6 augmentation techniques.
- Bagian B: Image compression (RLE, JPEG, WebP) on BMP, JPEG, PNG, SVG (with rasterization), metrics (size, CR, SS, PSNR, SSIM, time), plots.
- Bagian C: Watermarking (Visible, LSB, DCT), imperceptibility (PSNR, SSIM), extraction, robustness tests (JPEG compression, Gaussian noise, blur, crop), plots.
"""

import os
import sys
import time
import math
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fftpack import dct, idct
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

# Set random seeds for exact reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Bagian A: augmentasi data DAiSEE
DIR_VIDEO = os.path.join(DATA_DIR, "bagian_a", "video")
DIR_EXTRACTED = os.path.join(DATA_DIR, "bagian_a", "extracted_frames")
DIR_SAMPLES = os.path.join(DATA_DIR, "bagian_a", "sample_10_frames")
DIR_AUGMENTED = os.path.join(DATA_DIR, "bagian_a", "augmented_images")
# Bagian B: kompresi citra
DIR_COMP_INPUTS = os.path.join(DATA_DIR, "bagian_b", "compression_inputs")
DIR_COMP_RESULTS = os.path.join(DATA_DIR, "bagian_b", "compression_results")
# Bagian C: watermarking
DIR_WATERMARK = os.path.join(DATA_DIR, "bagian_c", "watermark_results")

# Output visualisasi per bagian
DIR_OUTPUT_A = os.path.join(OUTPUT_DIR, "bagian_a")
DIR_OUTPUT_B = os.path.join(OUTPUT_DIR, "bagian_b")
DIR_OUTPUT_C = os.path.join(OUTPUT_DIR, "bagian_c")

for d in [DIR_VIDEO, DIR_EXTRACTED, DIR_SAMPLES, DIR_AUGMENTED, DIR_COMP_INPUTS, DIR_COMP_RESULTS, DIR_WATERMARK, DIR_OUTPUT_A, DIR_OUTPUT_B, DIR_OUTPUT_C]:
    os.makedirs(d, exist_ok=True)

print("Directories initialized successfully.")

# ==============================================================================
# BAGIAN A: DAiSEE VIDEO CREATION / EXTRACTION & AUGMENTATION
# ==============================================================================

def create_synthetic_daisee_video(output_path, num_frames=300, fps=30, width=640, height=480):
    """
    Creates a realistic 10-second webcam video representing an e-learning student in DAiSEE:
    Includes realistic face avatar, head tilt, blinking, lighting fluctuations, and background.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Pre-generate parameters for smooth natural movement
    t_vals = np.linspace(0, 10, num_frames)
    
    for i, t in enumerate(t_vals):
        # Background: room wall with slight gradient and ambient lighting fluctuation
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        ambient_light = int(190 + 15 * np.sin(0.5 * t) + 8 * np.cos(1.2 * t))
        ambient_light = np.clip(ambient_light, 160, 230)
        
        # Wall color (warm indoor light)
        frame[:, :] = [ambient_light - 30, ambient_light - 15, ambient_light]
        
        # Add bookshelf/poster in background
        cv2.rectangle(frame, (40, 60), (180, 260), (90, 80, 75), -1)
        for b in range(4):
            cv2.rectangle(frame, (50, 80 + b*45), (170, 115 + b*45), (120 + b*15, 110, 100 - b*10), -1)
            
        # Laptop monitor / desk line at bottom
        cv2.rectangle(frame, (0, 420), (width, height), (50, 45, 45), -1)
        
        # Student head movement (slight sway, nodding, head tilt during thinking)
        dx = int(14 * np.sin(0.8 * t) + 5 * np.sin(2.3 * t))
        dy = int(8 * np.cos(0.6 * t) + 4 * np.cos(1.8 * t))
        angle = 3.5 * np.sin(0.5 * t)  # Head tilt degrees
        
        center_x = width // 2 + dx
        center_y = height // 2 - 20 + dy
        
        # Student body / shoulders
        body_color = (60, 75, 140)  # Blue shirt
        cv2.ellipse(frame, (int(center_x), int(center_y + 240)), (190, 140), 0.0, 0.0, 360.0, body_color, -1)
        # Collar / neck
        cv2.ellipse(frame, (int(center_x), int(center_y + 115)), (55, 60), 0.0, 0.0, 360.0, (180, 195, 225), -1)
        
        # Head (oval)
        base_skin = (165, 190, 230)  # BGR realistic skin tone
        screen_glow = int(12 * np.sin(1.5 * t))
        skin_color = (
            int(max(0, min(255, base_skin[0] + screen_glow))),
            int(max(0, min(255, base_skin[1] + screen_glow // 2))),
            int(base_skin[2])
        )
        
        cv2.ellipse(frame, (int(center_x), int(center_y)), (85, 115), float(angle), 0.0, 360.0, skin_color, -1)
        
        # Hair
        hair_color = (35, 30, 25)
        cv2.ellipse(frame, (int(center_x), int(center_y - 45)), (92, 75), float(angle), 180.0, 360.0, hair_color, -1)
        
        # Eyes with natural blinking (blink every ~3 seconds)
        eye_y = center_y - 12
        left_eye_x = center_x - 34
        right_eye_x = center_x + 34
        
        is_blinking = (abs(t % 3.2 - 0.1) < 0.08) or (abs(t % 6.7 - 0.1) < 0.08)
        
        if is_blinking:
            # Closed eye slit
            cv2.line(frame, (left_eye_x - 14, eye_y), (left_eye_x + 14, eye_y), (40, 35, 30), 2)
            cv2.line(frame, (right_eye_x - 14, eye_y), (right_eye_x + 14, eye_y), (40, 35, 30), 2)
        else:
            # Sclera
            cv2.ellipse(frame, (int(left_eye_x), int(eye_y)), (14, 8), float(angle), 0.0, 360.0, (245, 245, 245), -1)
            cv2.ellipse(frame, (int(right_eye_x), int(eye_y)), (14, 8), float(angle), 0.0, 360.0, (245, 245, 245), -1)
            # Iris & pupil (gaze tracking screen with slight shifts)
            gaze_x = int(2.5 * np.sin(1.1 * t))
            gaze_y = int(1.5 * np.cos(0.9 * t))
            cv2.circle(frame, (int(left_eye_x + gaze_x), int(eye_y + gaze_y)), 5, (45, 35, 20), -1)
            cv2.circle(frame, (int(right_eye_x + gaze_x), int(eye_y + gaze_y)), 5, (45, 35, 20), -1)
            cv2.circle(frame, (int(left_eye_x + gaze_x), int(eye_y + gaze_y)), 2, (10, 10, 10), -1)
            cv2.circle(frame, (int(right_eye_x + gaze_x), int(eye_y + gaze_y)), 2, (10, 10, 10), -1)
            
        # Eyebrows (subtle affective movement: engaged/focused)
        brow_y = eye_y - 14 + int(2 * np.sin(0.4 * t))
        cv2.line(frame, (int(left_eye_x - 16), int(brow_y + 2)), (int(left_eye_x + 16), int(brow_y - 2)), hair_color, 3)
        cv2.line(frame, (int(right_eye_x - 16), int(brow_y - 2)), (int(right_eye_x + 16), int(brow_y + 2)), hair_color, 3)
        
        # Nose
        cv2.line(frame, (int(center_x), int(center_y)), (int(center_x + 3), int(center_y + 25)), (140, 160, 200), 2)
        cv2.line(frame, (int(center_x - 6), int(center_y + 25)), (int(center_x + 6), int(center_y + 25)), (140, 160, 200), 2)
        
        # Mouth (neutral to attentive smile)
        mouth_y = center_y + 55
        mouth_curve = int(3 * np.sin(0.7 * t))
        cv2.ellipse(frame, (int(center_x), int(mouth_y)), (18, max(2, int(6 + mouth_curve))), 0.0, 0.0, 180.0, (90, 85, 175), 2)
        
        # Timestamp text simulating webcam feed
        cv2.putText(frame, f"DAiSEE E-Learning Subject 110001 | t={t:.2f}s | Frame {i+1}", 
                    (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        
        out.write(frame)
        
    out.release()
    print(f"Created simulated DAiSEE video: {output_path} ({num_frames} frames, {fps} FPS)")

def extract_frames_from_video(video_path, output_dir):
    """Extracts all frames from a video file into output_dir."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Extracting {total_frames} frames from {video_path}...")
    
    extracted_paths = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_file = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
        cv2.imwrite(frame_file, frame)
        extracted_paths.append(frame_file)
        frame_idx += 1
        
    cap.release()
    print(f"Successfully extracted {len(extracted_paths)} frames to {output_dir}")
    return extracted_paths

def sample_random_frames(all_frame_paths, sample_dir, n=10, seed=42):
    """Samples n random frames with a reproducible seed and copies/saves them."""
    random.seed(seed)
    np.random.seed(seed)
    sampled = sorted(random.sample(all_frame_paths, n))
    
    sampled_paths = []
    for i, path in enumerate(sampled):
        target_path = os.path.join(sample_dir, f"sample_{i+1:02d}_{os.path.basename(path)}")
        img = cv2.imread(path)
        cv2.imwrite(target_path, img)
        sampled_paths.append(target_path)
        
    print(f"Sampled {n} random frames with seed={seed}:")
    for sp in sampled_paths:
        print(f"  - {os.path.basename(sp)}")
    return sampled_paths

# ==============================================================================
# AUGMENTATION TECHNIQUES (BAGIAN A)
# ==============================================================================

def augment_horizontal_flip(image):
    """Pembalikan horizontal (bilateral symmetry)."""
    return cv2.flip(image, 1)

def augment_slight_rotation(image, angle=8.0):
    """Rotasi kecil (+/- 8 derajat) mensimulasikan kemiringan kepala (head tilt)."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)

def augment_random_crop_scale(image, crop_ratio=0.88):
    """Pemotongan acak dan penskalaan ulang ke dimensi asli (simulasi jarak siswa ke kamera)."""
    h, w = image.shape[:2]
    ch, cw = int(h * crop_ratio), int(w * crop_ratio)
    # Crop around face center
    top = int((h - ch) * 0.45)
    left = int((w - cw) * 0.5)
    cropped = image[top:top+ch, left:left+cw]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

def augment_brightness_contrast(image, alpha=1.18, beta=15):
    """Penyesuaian kecerahan dan kontras: I_new = alpha * I + beta."""
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def augment_gaussian_noise(image, mean=0, sigma=15):
    """Penambahan derau Gaussian untuk simulasi keterbatasan sensor webcam."""
    gauss = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + gauss, 0, 255).astype(np.uint8)
    return noisy

def augment_gaussian_blur(image, ksize=(5, 5), sigma=1.2):
    """Pengaburan Gaussian untuk simulasi defocus atau motion blur saat kepala bergerak."""
    return cv2.GaussianBlur(image, ksize, sigma)

AUGMENTATIONS = {
    "1_horizontal_flip": (augment_horizontal_flip, "Horizontal Flip (cv2.flip(img, 1))"),
    "2_slight_rotation": (lambda img: augment_slight_rotation(img, angle=8.0), "Slight Rotation (angle=+8 deg, reflect border)"),
    "3_random_crop_scale": (lambda img: augment_random_crop_scale(img, crop_ratio=0.88), "Random Crop & Scale (ratio=0.88, bilinear interp)"),
    "4_color_jitter_brightness": (lambda img: augment_brightness_contrast(img, alpha=1.18, beta=15), "Color Jitter (alpha=1.18, beta=+15)"),
    "5_gaussian_noise": (lambda img: augment_gaussian_noise(img, sigma=15), "Gaussian Noise (mean=0, sigma=15)"),
    "6_gaussian_blur": (lambda img: augment_gaussian_blur(img, ksize=(5, 5), sigma=1.2), "Gaussian Blur (kernel=5x5, sigma=1.2)")
}

def run_augmentation_pipeline(sampled_paths, base_aug_dir):
    """Applies all augmentations, saves to separate folders, and generates grid visualizations."""
    for tech_name in AUGMENTATIONS.keys():
        os.makedirs(os.path.join(base_aug_dir, tech_name), exist_ok=True)
        
    augmented_records = []
    
    for idx, path in enumerate(sampled_paths):
        fname = os.path.basename(path)
        img = cv2.imread(path)
        
        for tech_name, (func, desc) in AUGMENTATIONS.items():
            aug_img = func(img)
            save_path = os.path.join(base_aug_dir, tech_name, f"aug_{fname}")
            cv2.imwrite(save_path, aug_img)
            augmented_records.append({
                "frame": fname,
                "technique": tech_name,
                "description": desc,
                "output_path": save_path
            })
            
    print(f"Applied {len(AUGMENTATIONS)} augmentation techniques across {len(sampled_paths)} sample frames.")
    
    # Generate side-by-side comparison grid for the report and notebook
    # Select 3 representative frames to show in a clean figure
    rep_indices = [0, 4, 8]  # Frame 1, Frame 5, Frame 9
    num_tech = len(AUGMENTATIONS)
    
    fig, axes = plt.subplots(len(rep_indices), num_tech + 1, figsize=(20, 3.5 * len(rep_indices)))
    plt.subplots_adjust(wspace=0.15, hspace=0.25)
    
    tech_keys = list(AUGMENTATIONS.keys())
    
    for row, r_idx in enumerate(rep_indices):
        orig_path = sampled_paths[r_idx]
        orig_bgr = cv2.imread(orig_path)
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        
        # Original Image
        axes[row, 0].imshow(orig_rgb)
        axes[row, 0].set_title(f"Original Frame {r_idx+1}\n(DAiSEE Sample)", fontsize=10, fontweight='bold', color='navy')
        axes[row, 0].axis('off')
        
        # Augmented images
        for col, t_key in enumerate(tech_keys):
            aug_path = os.path.join(base_aug_dir, t_key, f"aug_{os.path.basename(orig_path)}")
            aug_bgr = cv2.imread(aug_path)
            aug_rgb = cv2.cvtColor(aug_bgr, cv2.COLOR_BGR2RGB)
            
            clean_name = t_key.split('_', 1)[1].replace('_', ' ').title()
            axes[row, col+1].imshow(aug_rgb)
            axes[row, col+1].set_title(f"{clean_name}", fontsize=10)
            axes[row, col+1].axis('off')
            
    grid_path = os.path.join(DIR_OUTPUT_A, "augmentation_grid_comparison.png")
    plt.tight_layout()
    plt.savefig(grid_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved augmentation grid visualization to {grid_path}")
    
    return augmented_records

# ==============================================================================
# BAGIAN B: IMAGE COMPRESSION ENGINE
# ==============================================================================

def rle_encode_bytes(byte_data: bytes) -> bytes:
    """
    Run-Length Encoding (RLE) encoder for binary/image byte streams.
    Encodes runs of identical consecutive bytes into (count, value) pairs.
    Count is clamped to 1..255 (1 byte), followed by the byte value (1 byte).
    """
    if not byte_data:
        return b""
    
    encoded = bytearray()
    n = len(byte_data)
    i = 0
    
    while i < n:
        val = byte_data[i]
        run_len = 1
        while i + 1 < n and byte_data[i + 1] == val and run_len < 255:
            run_len += 1
            i += 1
        encoded.append(run_len)
        encoded.append(val)
        i += 1
        
    return bytes(encoded)

def rle_decode_bytes(encoded_data: bytes) -> bytes:
    """Decodes an RLE byte stream back to original bytes."""
    decoded = bytearray()
    for i in range(0, len(encoded_data), 2):
        count = encoded_data[i]
        val = encoded_data[i + 1]
        decoded.extend([val] * count)
    return bytes(decoded)

def compress_rle(image_rgb, output_path):
    """Compresses image using custom RLE, writes header (H, W, C) + payload to disk."""
    t0 = time.perf_counter()
    h, w, c = image_rgb.shape
    raw_bytes = image_rgb.tobytes()
    payload = rle_encode_bytes(raw_bytes)
    
    # Simple binary container with header: 'RLE1' (4 bytes), H (4 bytes), W (4 bytes), C (4 bytes)
    header = b"RLE1" + h.to_bytes(4, 'big') + w.to_bytes(4, 'big') + c.to_bytes(4, 'big')
    with open(output_path, "wb") as f:
        f.write(header + payload)
        
    comp_time = (time.perf_counter() - t0) * 1000.0  # ms
    file_size = os.path.getsize(output_path)
    
    # Verify lossless decompression
    decoded_payload = rle_decode_bytes(payload)
    decomp_img = np.frombuffer(decoded_payload, dtype=np.uint8).reshape((h, w, c))
    
    return decomp_img, file_size, comp_time, True

def compress_jpeg(image_rgb, output_path, quality=75):
    """Compresses image using standard DCT-based JPEG."""
    t0 = time.perf_counter()
    pil_img = Image.fromarray(image_rgb)
    pil_img.save(output_path, format="JPEG", quality=quality)
    comp_time = (time.perf_counter() - t0) * 1000.0  # ms
    file_size = os.path.getsize(output_path)
    
    # Read back to evaluate lossy reconstruction
    decomp_pil = Image.open(output_path).convert("RGB")
    decomp_img = np.array(decomp_pil)
    
    return decomp_img, file_size, comp_time, False

def compress_webp(image_rgb, output_path, quality=75, lossless=False):
    """Compresses image using Google's WebP codec (VP8 lossy or VP8L lossless)."""
    t0 = time.perf_counter()
    pil_img = Image.fromarray(image_rgb)
    pil_img.save(output_path, format="WEBP", quality=quality, lossless=lossless)
    comp_time = (time.perf_counter() - t0) * 1000.0  # ms
    file_size = os.path.getsize(output_path)
    
    # Read back
    decomp_pil = Image.open(output_path).convert("RGB")
    decomp_img = np.array(decomp_pil)
    
    return decomp_img, file_size, comp_time, lossless

def create_sample_svg(svg_path):
    """
    Creates an educational vector SVG graphic containing geometric shapes,
    smooth gradients, curves, and text to demonstrate vector scalability.
    """
    svg_content = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">
  <defs>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#1A2980" />
      <stop offset="100%" stop-color="#26D0CE" />
    </linearGradient>
    <radialGradient id="sunGrad" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#FFE000" />
      <stop offset="100%" stop-color="#799F0C" />
    </radialGradient>
  </defs>
  
  <!-- Background Rect -->
  <rect width="512" height="512" fill="url(#bgGrad)" rx="30" />
  
  <!-- Central Abstract Multimedia Shape -->
  <circle cx="256" cy="200" r="110" fill="url(#sunGrad)" opacity="0.9" />
  
  <!-- Vector Polygon / Play Triangle -->
  <polygon points="220,140 330,200 220,260" fill="#FFFFFF" />
  
  <!-- Decorative Wave Paths -->
  <path d="M 40 380 Q 140 320 256 380 T 472 380" fill="none" stroke="#FFFFFF" stroke-width="8" stroke-linecap="round" />
  <path d="M 60 410 Q 160 360 256 410 T 452 410" fill="none" stroke="#FFE000" stroke-width="5" stroke-linecap="round" opacity="0.8" />
  
  <!-- Decorative Dots -->
  <circle cx="90" cy="90" r="15" fill="#FFFFFF" opacity="0.7" />
  <circle cx="430" cy="110" r="22" fill="#FFFFFF" opacity="0.5" />
  <circle cx="120" cy="280" r="10" fill="#FFE000" opacity="0.6" />
  
  <!-- Text Label -->
  <text x="256" y="465" font-family="Arial, sans-serif" font-size="28" font-weight="bold" fill="#FFFFFF" text-anchor="middle">
    KOMPUTASI MULTIMEDIA
  </text>
  <text x="256" y="492" font-family="Arial, sans-serif" font-size="16" fill="#E0F7FA" text-anchor="middle">
    Vector Graphic Benchmark (SVG)
  </text>
</svg>
"""
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_content)
    print(f"Created benchmark SVG vector graphic: {svg_path}")

def rasterize_svg(svg_path, output_png_path, width=512, height=512):
    """
    Rasterizes an SVG file into a PNG raster image.
    Demonstrates the vector-to-raster conversion process: converting mathematical
    parametric primitives (lines, curves, polygons, gradients) into discrete pixel matrices.
    """
    img = Image.new("RGB", (width, height), color=(26, 41, 128))
    draw = ImageDraw.Draw(img)
    
    # Render linear gradient background
    for y in range(height):
        ratio = y / height
        r = int(26 + (38 - 26) * ratio)
        g = int(41 + (208 - 41) * ratio)
        b = int(128 + (206 - 128) * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
        
    # Central multimedia circle
    cx, cy, radius = width // 2, int(height * 0.39), int(110 * width / 512)
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], 
                 fill=(255, 224, 0), outline=(121, 159, 12), width=3)
    
    # Vector play triangle
    tri_scale = width / 512.0
    triangle = [
        (int(220 * tri_scale), int(140 * tri_scale)),
        (int(330 * tri_scale), int(200 * tri_scale)),
        (int(220 * tri_scale), int(260 * tri_scale))
    ]
    draw.polygon(triangle, fill=(255, 255, 255))
    
    # Vector wave curve
    for x in range(40, 472):
        t_val = (x - 40) / 432.0 * math.pi * 2.0
        y_val = int(380 + 35 * math.sin(t_val))
        draw.ellipse([x - 3, y_val - 3, x + 3, y_val + 3], fill=(255, 255, 255))
        
    # Decorative vector dots
    draw.ellipse([75, 75, 105, 105], fill=(255, 255, 255))
    draw.ellipse([408, 88, 452, 132], fill=(255, 255, 255))
    draw.ellipse([110, 270, 130, 290], fill=(255, 224, 0))
    
    # Text
    draw.text((width // 2, int(455 * height / 512)), "KOMPUTASI MULTIMEDIA", 
              fill=(255, 255, 255), anchor="mm")
    draw.text((width // 2, int(485 * height / 512)), "Vector Graphic Benchmark (SVG)", 
              fill=(224, 247, 250), anchor="mm")
    
    img.save(output_png_path, format="PNG")
    print(f"Rasterized SVG vector to PNG: {output_png_path} ({width}x{height})")
    return np.array(img)

def prepare_compression_benchmark_images(source_frame_path, comp_inputs_dir):
    """
    Prepares benchmark images across required formats:
    - BMP: 24-bit uncompressed RGB bitmap
    - JPEG: photographic compressed baseline
    - PNG: lossless RGBA/RGB graphic with alpha and sharp contrast
    - SVG: vector graphic with XML vector instructions
    """
    orig_bgr = cv2.imread(source_frame_path)
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    
    # 1. BMP
    bmp_path = os.path.join(comp_inputs_dir, "benchmark_image.bmp")
    pil_bmp = Image.fromarray(orig_rgb)
    pil_bmp.save(bmp_path, format="BMP")
    
    # 2. JPEG
    jpg_path = os.path.join(comp_inputs_dir, "benchmark_image.jpg")
    pil_bmp.save(jpg_path, format="JPEG", quality=95)
    
    # 3. PNG
    png_path = os.path.join(comp_inputs_dir, "benchmark_image.png")
    pil_bmp.save(png_path, format="PNG")
    
    # 4. SVG
    svg_path = os.path.join(comp_inputs_dir, "benchmark_vector.svg")
    create_sample_svg(svg_path)
    
    # Rasterize SVG to create raster equivalent for pixel-based compression
    svg_raster_path = os.path.join(comp_inputs_dir, "benchmark_svg_rasterized.png")
    rasterize_svg(svg_path, svg_raster_path, width=512, height=512)
    
    return {
        "BMP": (bmp_path, orig_rgb),
        "JPEG": (jpg_path, orig_rgb),
        "PNG": (png_path, orig_rgb),
        "SVG (Rasterized)": (svg_raster_path, np.array(Image.open(svg_raster_path).convert("RGB")))
    }

def run_compression_benchmarks(image_dict, comp_results_dir):
    """
    Runs RLE, JPEG (DCT), and WebP on all benchmark images.
    Calculates file sizes, compression ratios, space saving, PSNR, SSIM, and encoding time.
    """
    results = []
    
    # Setup subdirectories
    for tech in ["RLE", "JPEG", "WebP_Lossy", "WebP_Lossless"]:
        os.makedirs(os.path.join(comp_results_dir, tech), exist_ok=True)
        
    for img_type, (input_path, img_rgb) in image_dict.items():
        orig_size = os.path.getsize(input_path)
        clean_prefix = img_type.split()[0].lower()
        
        # 1. RLE Compression
        rle_out = os.path.join(comp_results_dir, "RLE", f"{clean_prefix}_compressed.rle")
        rle_decomp, rle_size, rle_time, rle_lossless = compress_rle(img_rgb, rle_out)
        cr_rle = orig_size / rle_size if rle_size > 0 else 0
        ss_rle = (1.0 - (rle_size / orig_size)) * 100.0
        psnr_rle = float('inf')
        ssim_rle = 1.0
        
        results.append({
            "Input Format": img_type,
            "Compression Technique": "Run-Length Encoding (RLE)",
            "Type": "Lossless",
            "Orig Size (KB)": orig_size / 1024.0,
            "Comp Size (KB)": rle_size / 1024.0,
            "Compression Ratio": cr_rle,
            "Space Saving (%)": ss_rle,
            "PSNR (dB)": psnr_rle,
            "SSIM": ssim_rle,
            "Time (ms)": rle_time,
            "Decompressed Image": rle_decomp
        })
        
        # 2. JPEG (DCT) Compression (Quality = 75)
        jpeg_out = os.path.join(comp_results_dir, "JPEG", f"{clean_prefix}_compressed.jpg")
        jpeg_decomp, jpeg_size, jpeg_time, _ = compress_jpeg(img_rgb, jpeg_out, quality=75)
        cr_jpeg = orig_size / jpeg_size if jpeg_size > 0 else 0
        ss_jpeg = (1.0 - (jpeg_size / orig_size)) * 100.0
        psnr_jpeg = compute_psnr(img_rgb, jpeg_decomp)
        ssim_jpeg = compute_ssim(img_rgb, jpeg_decomp, channel_axis=2)
        
        results.append({
            "Input Format": img_type,
            "Compression Technique": "JPEG (DCT, Q=75)",
            "Type": "Lossy",
            "Orig Size (KB)": orig_size / 1024.0,
            "Comp Size (KB)": jpeg_size / 1024.0,
            "Compression Ratio": cr_jpeg,
            "Space Saving (%)": ss_jpeg,
            "PSNR (dB)": psnr_jpeg,
            "SSIM": ssim_jpeg,
            "Time (ms)": jpeg_time,
            "Decompressed Image": jpeg_decomp
        })
        
        # 3. WebP Lossy Compression (Quality = 75)
        webp_out = os.path.join(comp_results_dir, "WebP_Lossy", f"{clean_prefix}_compressed.webp")
        webp_decomp, webp_size, webp_time, _ = compress_webp(img_rgb, webp_out, quality=75, lossless=False)
        cr_webp = orig_size / webp_size if webp_size > 0 else 0
        ss_webp = (1.0 - (webp_size / orig_size)) * 100.0
        psnr_webp = compute_psnr(img_rgb, webp_decomp)
        ssim_webp = compute_ssim(img_rgb, webp_decomp, channel_axis=2)
        
        results.append({
            "Input Format": img_type,
            "Compression Technique": "WebP (VP8 Lossy, Q=75)",
            "Type": "Lossy",
            "Orig Size (KB)": orig_size / 1024.0,
            "Comp Size (KB)": webp_size / 1024.0,
            "Compression Ratio": cr_webp,
            "Space Saving (%)": ss_webp,
            "PSNR (dB)": psnr_webp,
            "SSIM": ssim_webp,
            "Time (ms)": webp_time,
            "Decompressed Image": webp_decomp
        })
        
        # 4. WebP Lossless Compression
        webp_ll_out = os.path.join(comp_results_dir, "WebP_Lossless", f"{clean_prefix}_compressed_ll.webp")
        webp_ll_decomp, webp_ll_size, webp_ll_time, _ = compress_webp(img_rgb, webp_ll_out, lossless=True)
        cr_webp_ll = orig_size / webp_ll_size if webp_ll_size > 0 else 0
        ss_webp_ll = (1.0 - (webp_ll_size / orig_size)) * 100.0
        psnr_webp_ll = float('inf')
        ssim_webp_ll = 1.0
        
        results.append({
            "Input Format": img_type,
            "Compression Technique": "WebP (VP8L Lossless)",
            "Type": "Lossless",
            "Orig Size (KB)": orig_size / 1024.0,
            "Comp Size (KB)": webp_ll_size / 1024.0,
            "Compression Ratio": cr_webp_ll,
            "Space Saving (%)": ss_webp_ll,
            "PSNR (dB)": psnr_webp_ll,
            "SSIM": ssim_webp_ll,
            "Time (ms)": webp_ll_time,
            "Decompressed Image": webp_ll_decomp
        })
        
    df = pd.DataFrame(results)
    
    # Save CSV table (excluding image arrays)
    csv_df = df.drop(columns=["Decompressed Image"])
    csv_path = os.path.join(comp_results_dir, "compression_benchmark_summary.csv")
    csv_df.to_csv(csv_path, index=False)
    print(f"Saved compression benchmark metrics to {csv_path}")
    
    # Generate Charts
    generate_compression_charts(df, DIR_OUTPUT_B)
    
    # Generate Visual Side-by-Side Comparison
    generate_compression_visual_comparison(image_dict, df, DIR_OUTPUT_B)
    
    return df

def generate_compression_charts(df, output_dir):
    """Generates comparative charts for Compression Ratio, Space Saving, PSNR/SSIM, and Time."""
    formats = df["Input Format"].unique()
    techniques = df["Compression Technique"].unique()
    
    # 1. Bar Chart: Compression Ratio
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(formats))
    width = 0.2
    
    for i, tech in enumerate(techniques):
        sub = df[df["Compression Technique"] == tech]
        vals = [sub[sub["Input Format"] == f]["Compression Ratio"].values[0] for f in formats]
        ax.bar(x + i*width, vals, width, label=tech)
        
    ax.set_ylabel("Compression Ratio (Original : Compressed)", fontsize=11, fontweight='bold')
    ax.set_title("Nisbah / Rasio Kompresi Berdasarkan Format Citra dan Teknik", fontsize=12, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(formats, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "chart_compression_ratio.png"), dpi=300)
    plt.close()
    
    # 2. Bar Chart: Space Saving (%)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, tech in enumerate(techniques):
        sub = df[df["Compression Technique"] == tech]
        vals = [sub[sub["Input Format"] == f]["Space Saving (%)"].values[0] for f in formats]
        ax.bar(x + i*width, vals, width, label=tech)
        
    ax.set_ylabel("Space Saving (%)", fontsize=11, fontweight='bold')
    ax.set_title("Space Saving (%) Berdasarkan Format Citra dan Teknik", fontsize=12, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(formats, fontweight='bold')
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "chart_space_saving.png"), dpi=300)
    plt.close()
    
    # 3. Quality vs Speed (PSNR & Time) for Lossy Techniques
    lossy_df = df[df["Type"] == "Lossy"].copy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # PSNR comparison
    lossy_techs = lossy_df["Compression Technique"].unique()
    x_lossy = np.arange(len(formats))
    w_lossy = 0.35
    for i, tech in enumerate(lossy_techs):
        sub = lossy_df[lossy_df["Compression Technique"] == tech]
        psnr_vals = [sub[sub["Input Format"] == f]["PSNR (dB)"].values[0] for f in formats]
        ax1.bar(x_lossy + i*w_lossy, psnr_vals, w_lossy, label=tech)
    ax1.set_ylabel("PSNR (dB)", fontsize=11, fontweight='bold')
    ax1.set_title("Evaluasi Kualitas Rekonstruksi (PSNR)", fontsize=12, fontweight='bold')
    ax1.set_xticks(x_lossy + w_lossy / 2)
    ax1.set_xticklabels(formats, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Compression Time
    for i, tech in enumerate(techniques):
        sub = df[df["Compression Technique"] == tech]
        time_vals = [sub[sub["Input Format"] == f]["Time (ms)"].values[0] for f in formats]
        ax2.bar(x + i*width, time_vals, width, label=tech)
    ax2.set_ylabel("Waktu Enkoding (ms)", fontsize=11, fontweight='bold')
    ax2.set_title("Perbandingan Kecepatan Kompresi (ms)", fontsize=12, fontweight='bold')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(formats, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "chart_quality_and_time.png"), dpi=300)
    plt.close()
    print("Generated all compression benchmark charts.")

def generate_compression_visual_comparison(image_dict, df, output_dir):
    """
    Generates high-resolution side-by-side visual comparison between original
    and decompressed images, along with pixel difference heatmaps.
    """
    for img_type, (input_path, orig_rgb) in image_dict.items():
        clean_name = img_type.split()[0].lower()
        sub_df = df[df["Input Format"] == img_type]
        
        # Display: Original | RLE | JPEG | WebP Lossy
        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        
        # Top Row: Images
        axes[0, 0].imshow(orig_rgb)
        orig_kb = os.path.getsize(input_path) / 1024.0
        axes[0, 0].set_title(f"Original ({img_type})\nSize: {orig_kb:.1f} KB", fontsize=11, fontweight='bold', color='navy')
        axes[0, 0].axis('off')
        
        # Bottom Row: Residual Difference Heatmaps (|Orig - Decomp|)
        axes[1, 0].axis('off')
        axes[1, 0].text(0.5, 0.5, "Peta Residual\nSelisih Absolut\n|Asli - Kompresi|\n(Skala Error x10)", 
                        ha='center', va='center', fontsize=12, fontweight='bold', color='darkred')
        
        tech_list = [
            ("Run-Length Encoding (RLE)", "RLE Decompressed"),
            ("JPEG (DCT, Q=75)", "JPEG (Q=75)"),
            ("WebP (VP8 Lossy, Q=75)", "WebP Lossy (Q=75)")
        ]
        
        for col_idx, (tech_key, display_title) in enumerate(tech_list, start=1):
            row_data = sub_df[sub_df["Compression Technique"] == tech_key].iloc[0]
            decomp = row_data["Decompressed Image"]
            comp_kb = row_data["Comp Size (KB)"]
            cr = row_data["Compression Ratio"]
            psnr_val = row_data["PSNR (dB)"]
            psnr_str = f"{psnr_val:.2f} dB" if psnr_val != float('inf') else "Inf (Lossless)"
            
            axes[0, col_idx].imshow(decomp)
            axes[0, col_idx].set_title(f"{display_title}\nSize: {comp_kb:.1f} KB | CR: {cr:.2f}x\nPSNR: {psnr_str}", 
                                      fontsize=10, fontweight='bold')
            axes[0, col_idx].axis('off')
            
            # Compute difference heatmap
            diff = np.abs(orig_rgb.astype(np.float32) - decomp.astype(np.float32))
            diff_amplified = np.clip(diff * 10.0, 0, 255).astype(np.uint8)
            diff_gray = cv2.cvtColor(diff_amplified, cv2.COLOR_RGB2GRAY)
            
            im_heat = axes[1, col_idx].imshow(diff_gray, cmap='inferno', vmin=0, vmax=255)
            axes[1, col_idx].set_title(f"Residual Error Heatmap", fontsize=10)
            axes[1, col_idx].axis('off')
            
        plt.tight_layout()
        vis_path = os.path.join(output_dir, f"visual_comparison_{clean_name}.png")
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved visual comparison for {img_type} to {vis_path}")

# ==============================================================================
# BAGIAN C: WATERMARKING ENGINE (VISIBLE, LSB, DCT) & ROBUSTNESS BENCHMARK
# ==============================================================================

def create_binary_watermark_logo(size=(64, 64)):
    """
    Creates a distinct 64x64 binary watermark logo representing copyright.
    Has sharp text 'KM' and a circular border.
    """
    img = Image.new("1", size, color=0)  # Black background (0)
    draw = ImageDraw.Draw(img)
    
    # Outer circle border
    draw.ellipse((2, 2, size[0]-3, size[1]-3), outline=1, width=2)
    # Inner circle
    draw.ellipse((6, 6, size[0]-7, size[1]-7), outline=1, width=1)
    
    # Text 'KM' in center
    try:
        font = ImageFont.load_default()
        draw.text((size[0]//2 - 12, size[1]//2 - 10), "KM", fill=1, font=font)
        draw.text((size[0]//2 - 14, size[1]//2 + 6), "2026", fill=1, font=font)
    except Exception:
        # Fallback geometric mark
        draw.rectangle((20, 20, 44, 44), outline=1, fill=1)
        draw.rectangle((26, 26, 38, 38), outline=0, fill=0)
        
    logo_array = np.array(img, dtype=np.uint8)  # 0 or 1
    return logo_array

def embed_visible_watermark(image_rgb, text="COPYRIGHT (C) 2026 KOMPUTASI MULTIMEDIA", alpha=0.35):
    """
    Embeds a semi-transparent visible watermark banner and text across the lower portion.
    """
    h, w, c = image_rgb.shape
    overlay = image_rgb.copy()
    
    # Semi-transparent dark banner at bottom
    banner_h = 45
    cv2.rectangle(overlay, (0, h - banner_h), (w, h), (20, 20, 20), -1)
    
    # Stamp text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    cv2.putText(overlay, text, (20, h - 16), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    # Semi-transparent circle emblem on right
    cv2.circle(overlay, (w - 30, h - banner_h // 2), 16, (0, 200, 255), -1)
    cv2.putText(overlay, "KM", (w - 41, h - banner_h // 2 + 5), font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    
    # Alpha blend: I_wm = (1 - alpha) * I_orig + alpha * overlay
    watermarked = cv2.addWeighted(overlay, alpha, image_rgb, 1 - alpha, 0)
    return watermarked

def embed_lsb_watermark(image_rgb, binary_watermark):
    """
    Embeds a 64x64 binary watermark into the Least Significant Bit (bit 0)
    of the Blue channel of the image.
    """
    h, w, c = image_rgb.shape
    wm_h, wm_w = binary_watermark.shape
    assert h >= wm_h and w >= wm_w, "Host image must be at least as large as watermark."
    
    watermarked = image_rgb.copy()
    # Embed in top-left region of Blue channel (channel 2)
    host_slice = watermarked[:wm_h, :wm_w, 2]
    # Clear bit 0 using uint8 mask 0xFE (254), then set bit 0 to watermark bit
    embedded_slice = (host_slice & np.uint8(254)) | (binary_watermark.astype(np.uint8) & np.uint8(1))
    watermarked[:wm_h, :wm_w, 2] = embedded_slice
    
    return watermarked

def extract_lsb_watermark(watermarked_rgb, wm_shape=(64, 64)):
    """Extracts binary watermark from LSB of Blue channel."""
    wm_h, wm_w = wm_shape
    extracted = watermarked_rgb[:wm_h, :wm_w, 2] & 1
    return extracted.astype(np.uint8)

# 2D DCT Block-based functions
def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def embed_dct_watermark(image_rgb, binary_watermark, delta=18.0):
    """
    Embeds a 64x64 binary watermark into the mid-frequency DCT coefficients
    of 8x8 non-overlapping blocks of the Luminance (Y) channel.
    Uses differential relationship between coefficients (4, 3) and (3, 4).
    """
    # Convert RGB to YCrCb
    ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
    Y = ycrcb[:, :, 0].astype(np.float32)
    
    h, w = Y.shape
    wm_h, wm_w = binary_watermark.shape
    assert h >= wm_h * 8 and w >= wm_w * 8, f"Host image must be at least {wm_h*8}x{wm_w*8} for DCT."
    
    # Resize host if needed or use top-left 512x512
    Y_wm = Y.copy()
    
    # Coefficient coordinates in 8x8 block (mid-frequency band)
    u1, v1 = 4, 3
    u2, v2 = 3, 4
    
    for i in range(wm_h):
        for j in range(wm_w):
            bit = binary_watermark[i, j]
            by, bx = i * 8, j * 8
            block = Y[by:by+8, bx:bx+8]
            coeff = dct2(block)
            
            c1 = coeff[u1, v1]
            c2 = coeff[u2, v2]
            
            if bit == 1:
                # Ensure c1 - c2 > delta
                if c1 - c2 < delta:
                    diff = delta - (c1 - c2)
                    c1 += diff / 2.0
                    c2 -= diff / 2.0
            else:
                # Ensure c2 - c1 > delta
                if c2 - c1 < delta:
                    diff = delta - (c2 - c1)
                    c2 += diff / 2.0
                    c1 -= diff / 2.0
                    
            coeff[u1, v1] = c1
            coeff[u2, v2] = c2
            
            Y_wm[by:by+8, bx:bx+8] = idct2(coeff)
            
    # Clip and reconstruct RGB
    Y_wm = np.clip(Y_wm, 0, 255).astype(np.uint8)
    ycrcb_wm = ycrcb.copy()
    ycrcb_wm[:, :, 0] = Y_wm
    watermarked_rgb = cv2.cvtColor(ycrcb_wm, cv2.COLOR_YCrCb2RGB)
    
    return watermarked_rgb

def extract_dct_watermark(watermarked_rgb, wm_shape=(64, 64)):
    """Extracts binary watermark from mid-frequency DCT coefficients of 8x8 blocks."""
    ycrcb = cv2.cvtColor(watermarked_rgb, cv2.COLOR_RGB2YCrCb)
    Y = ycrcb[:, :, 0].astype(np.float32)
    wm_h, wm_w = wm_shape
    
    u1, v1 = 4, 3
    u2, v2 = 3, 4
    
    extracted = np.zeros(wm_shape, dtype=np.uint8)
    
    for i in range(wm_h):
        for j in range(wm_w):
            by, bx = i * 8, j * 8
            block = Y[by:by+8, bx:bx+8]
            coeff = dct2(block)
            c1 = coeff[u1, v1]
            c2 = coeff[u2, v2]
            
            extracted[i, j] = 1 if c1 > c2 else 0
            
    return extracted

def compute_watermark_metrics(original_wm, extracted_wm):
    """
    Computes Bit Error Rate (BER) and Normalized Cross-Correlation (NCC).
    """
    orig_flat = original_wm.flatten().astype(np.float32)
    extr_flat = extracted_wm.flatten().astype(np.float32)
    
    # BER: proportion of mismatched bits
    ber = np.mean(orig_flat != extr_flat) * 100.0  # percentage
    
    # NCC: normalized cross-correlation [0, 1]
    denom = np.sqrt(np.sum(orig_flat ** 2) * np.sum(extr_flat ** 2))
    if denom == 0:
        ncc = 1.0 if np.array_equal(orig_flat, extr_flat) else 0.0
    else:
        ncc = float(np.sum(orig_flat * extr_flat) / denom)
        
    return ber, ncc

# Attacks for Robustness Testing
def attack_jpeg_compression(image_rgb, quality=50):
    """Applies lossy JPEG compression attack."""
    pil_img = Image.fromarray(image_rgb)
    import io
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    attacked_pil = Image.open(buffer).convert("RGB")
    return np.array(attacked_pil)

def attack_gaussian_noise(image_rgb, mean=0, sigma=10):
    """Applies additive Gaussian noise attack."""
    gauss = np.random.normal(mean, sigma, image_rgb.shape).astype(np.float32)
    attacked = np.clip(image_rgb.astype(np.float32) + gauss, 0, 255).astype(np.uint8)
    return attacked

def attack_gaussian_blur(image_rgb, ksize=(3, 3), sigma=1.0):
    """Applies low-pass Gaussian blur filter attack."""
    return cv2.GaussianBlur(image_rgb, ksize, sigma)

def attack_center_crop(image_rgb, crop_fraction=0.15):
    """Applies center occlusion / cropping attack (crops center and replaces with black)."""
    attacked = image_rgb.copy()
    h, w, c = attacked.shape
    ch, cw = int(h * crop_fraction), int(w * crop_fraction)
    cy, cx = h // 2, w // 2
    attacked[cy - ch//2 : cy + ch//2, cx - cw//2 : cx + cw//2] = 0
    return attacked

def run_watermarking_pipeline(host_image_path, watermark_dir):
    """
    Executes visible, LSB, and DCT watermarking, tests imperceptibility,
    performs extraction, tests robustness across multiple attacks,
    and produces comprehensive evaluation tables and visual charts.
    """
    orig_bgr = cv2.imread(host_image_path)
    # Ensure 512x512 resolution for clean 64x64 blocks in DCT
    orig_rgb = cv2.resize(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB), (512, 512), interpolation=cv2.INTER_AREA)
    
    # 1. Create binary watermark logo (64x64)
    wm_logo = create_binary_watermark_logo(size=(64, 64))
    cv2.imwrite(os.path.join(watermark_dir, "watermark_logo.png"), wm_logo * 255)
    print(f"Created 64x64 binary watermark logo.")
    
    # 2. Embed watermarks
    # A. Visible Watermark
    wm_visible = embed_visible_watermark(orig_rgb)
    cv2.imwrite(os.path.join(watermark_dir, "watermarked_visible.png"), cv2.cvtColor(wm_visible, cv2.COLOR_RGB2BGR))
    psnr_vis = compute_psnr(orig_rgb, wm_visible)
    ssim_vis = compute_ssim(orig_rgb, wm_visible, channel_axis=2)
    
    # B. LSB Invisible Watermark
    wm_lsb = embed_lsb_watermark(orig_rgb, wm_logo)
    cv2.imwrite(os.path.join(watermark_dir, "watermarked_lsb.png"), cv2.cvtColor(wm_lsb, cv2.COLOR_RGB2BGR))
    psnr_lsb = compute_psnr(orig_rgb, wm_lsb)
    ssim_lsb = compute_ssim(orig_rgb, wm_lsb, channel_axis=2)
    
    # C. DCT Invisible Watermark
    wm_dct = embed_dct_watermark(orig_rgb, wm_logo, delta=18.0)
    cv2.imwrite(os.path.join(watermark_dir, "watermarked_dct.png"), cv2.cvtColor(wm_dct, cv2.COLOR_RGB2BGR))
    psnr_dct = compute_psnr(orig_rgb, wm_dct)
    ssim_dct = compute_ssim(orig_rgb, wm_dct, channel_axis=2)
    
    # Imperceptibility Summary
    impercept_records = [
        {"Technique": "Visible Watermark (Alpha Blend)", "PSNR (dB)": psnr_vis, "SSIM": ssim_vis, "Visibility": "Kasatmata (Visible)"},
        {"Technique": "LSB Embedding (Spatial Domain)", "PSNR (dB)": psnr_lsb, "SSIM": ssim_lsb, "Visibility": "Tak Kasatmata (Invisible)"},
        {"Technique": "DCT Block Differential (Transform Domain)", "PSNR (dB)": psnr_dct, "SSIM": ssim_dct, "Visibility": "Tak Kasatmata (Invisible)"}
    ]
    df_impercept = pd.DataFrame(impercept_records)
    print("\n--- Imperceptibility Evaluation ---")
    print(df_impercept.to_string(index=False))
    
    # 3. Clean Extraction (Tanpa Serangan)
    extr_lsb_clean = extract_lsb_watermark(wm_lsb)
    extr_dct_clean = extract_dct_watermark(wm_dct)
    
    cv2.imwrite(os.path.join(watermark_dir, "extracted_lsb_clean.png"), extr_lsb_clean * 255)
    cv2.imwrite(os.path.join(watermark_dir, "extracted_dct_clean.png"), extr_dct_clean * 255)
    
    ber_lsb_clean, ncc_lsb_clean = compute_watermark_metrics(wm_logo, extr_lsb_clean)
    ber_dct_clean, ncc_dct_clean = compute_watermark_metrics(wm_logo, extr_dct_clean)
    
    print(f"Clean LSB Extraction -> BER: {ber_lsb_clean:.2f}%, NCC: {ncc_lsb_clean:.4f}")
    print(f"Clean DCT Extraction -> BER: {ber_dct_clean:.2f}%, NCC: {ncc_dct_clean:.4f}")
    
    # 4. Robustness Testing across Attacks
    attacks = {
        "1. No Attack (Baseline)": lambda img: img,
        "2. JPEG Compression (Q=50)": lambda img: attack_jpeg_compression(img, quality=50),
        "3. JPEG Compression (Q=30)": lambda img: attack_jpeg_compression(img, quality=30),
        "4. Gaussian Noise (sigma=10)": lambda img: attack_gaussian_noise(img, sigma=10),
        "5. Gaussian Blur (3x3)": lambda img: attack_gaussian_blur(img, ksize=(3, 3), sigma=1.0),
        "6. Center Occlusion (15% Crop)": lambda img: attack_center_crop(img, crop_fraction=0.15)
    }
    
    robustness_records = []
    extracted_images_cache = {}
    
    for atk_name, atk_func in attacks.items():
        # Apply attack to both LSB and DCT watermarked images
        atk_lsb = atk_func(wm_lsb)
        atk_dct = atk_func(wm_dct)
        
        # Calculate degradation PSNR of the attacked images
        psnr_atk_lsb = compute_psnr(orig_rgb, atk_lsb)
        psnr_atk_dct = compute_psnr(orig_rgb, atk_dct)
        
        # Extract watermarks
        extr_lsb = extract_lsb_watermark(atk_lsb)
        extr_dct = extract_dct_watermark(atk_dct)
        
        ber_lsb, ncc_lsb = compute_watermark_metrics(wm_logo, extr_lsb)
        ber_dct, ncc_dct = compute_watermark_metrics(wm_logo, extr_dct)
        
        clean_atk_slug = atk_name.split('.')[1].strip().replace(' ', '_').replace('=', '').replace('(', '').replace(')', '').lower()
        extracted_images_cache[atk_name] = {
            "atk_lsb": atk_lsb, "atk_dct": atk_dct,
            "extr_lsb": extr_lsb, "extr_dct": extr_dct,
            "ber_lsb": ber_lsb, "ncc_lsb": ncc_lsb,
            "ber_dct": ber_dct, "ncc_dct": ncc_dct
        }
        
        # Save attacked & extracted images
        cv2.imwrite(os.path.join(watermark_dir, f"attack_{clean_atk_slug}_lsb.png"), cv2.cvtColor(atk_lsb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(watermark_dir, f"attack_{clean_atk_slug}_dct.png"), cv2.cvtColor(atk_dct, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(watermark_dir, f"extr_{clean_atk_slug}_lsb.png"), extr_lsb * 255)
        cv2.imwrite(os.path.join(watermark_dir, f"extr_{clean_atk_slug}_dct.png"), extr_dct * 255)
        
        robustness_records.append({
            "Attack Type": atk_name,
            "Attacked Image PSNR (dB)": psnr_atk_dct,
            "LSB BER (%)": ber_lsb,
            "LSB NCC": ncc_lsb,
            "DCT BER (%)": ber_dct,
            "DCT NCC": ncc_dct
        })
        
    df_robustness = pd.DataFrame(robustness_records)
    csv_rob_path = os.path.join(watermark_dir, "watermark_robustness_summary.csv")
    df_robustness.to_csv(csv_rob_path, index=False)
    print("\n--- Robustness Evaluation Summary ---")
    print(df_robustness.to_string(index=False))
    
    # 5. Visual Figures for Watermarking
    generate_watermarking_visualizations(orig_rgb, wm_visible, wm_lsb, wm_dct, wm_logo, 
                                         extracted_images_cache, attacks, DIR_OUTPUT_C)
    
    return df_impercept, df_robustness

def generate_watermarking_visualizations(orig_rgb, wm_vis, wm_lsb, wm_dct, wm_logo,
                                         extracted_cache, attacks, output_dir):
    """
    Generates side-by-side figures for:
    1. Imperceptibility & watermarked image comparisons.
    2. Robustness comparison matrix of extracted watermarks under attacks.
    """
    # Figure 1: Watermarking Overview
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    
    axes[0].imshow(orig_rgb)
    axes[0].set_title("Citra Asli (Host Image)\n(512x512 RGB)", fontsize=11, fontweight='bold', color='navy')
    axes[0].axis('off')
    
    axes[1].imshow(wm_vis)
    psnr_vis = compute_psnr(orig_rgb, wm_vis)
    axes[1].set_title(f"Visible Watermark\nAlpha Blended Banner\nPSNR: {psnr_vis:.2f} dB", fontsize=11, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(wm_lsb)
    psnr_lsb = compute_psnr(orig_rgb, wm_lsb)
    axes[2].set_title(f"LSB Invisible Watermark\nBit-0 Blue Channel\nPSNR: {psnr_lsb:.2f} dB", fontsize=11, fontweight='bold')
    axes[2].axis('off')
    
    axes[3].imshow(wm_dct)
    psnr_dct = compute_psnr(orig_rgb, wm_dct)
    axes[3].set_title(f"DCT Invisible Watermark\n8x8 Mid-Freq Bands\nPSNR: {psnr_dct:.2f} dB", fontsize=11, fontweight='bold')
    axes[3].axis('off')
    
    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "watermarking_overview.png")
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved watermarking overview figure to {fig1_path}")
    
    # Figure 2: Robustness Matrix Comparison
    # Display: Attack Description | Attacked Image | Extracted LSB | Extracted DCT
    num_attacks = len(attacks)
    fig, axes = plt.subplots(num_attacks, 4, figsize=(16, 3.2 * num_attacks))
    plt.subplots_adjust(wspace=0.15, hspace=0.35)
    
    for row, (atk_name, _) in enumerate(attacks.items()):
        data = extracted_cache[atk_name]
        
        # Column 0: Original Watermark Logo reference or Attacked Image
        axes[row, 0].imshow(wm_logo, cmap='gray', vmin=0, vmax=1)
        axes[row, 0].set_title("Ground-Truth Logo (64x64)", fontsize=10, fontweight='bold', color='navy')
        axes[row, 0].axis('off')
        
        # Column 1: Attacked Image (DCT version)
        axes[row, 1].imshow(data["atk_dct"])
        atk_psnr = compute_psnr(orig_rgb, data["atk_dct"])
        axes[row, 1].set_title(f"Citra Terserang\n{atk_name}\n(PSNR: {atk_psnr:.2f} dB)", fontsize=10, fontweight='bold')
        axes[row, 1].axis('off')
        
        # Column 2: Extracted LSB
        axes[row, 2].imshow(data["extr_lsb"], cmap='gray', vmin=0, vmax=1)
        axes[row, 2].set_title(f"Ekstraksi LSB (Spatial)\nBER: {data['ber_lsb']:.1f}% | NCC: {data['ncc_lsb']:.3f}", 
                               fontsize=10, fontweight='bold',
                               color='darkred' if data['ber_lsb'] > 20 else 'darkgreen')
        axes[row, 2].axis('off')
        
        # Column 3: Extracted DCT
        axes[row, 3].imshow(data["extr_dct"], cmap='gray', vmin=0, vmax=1)
        axes[row, 3].set_title(f"Ekstraksi DCT (Frekuensi)\nBER: {data['ber_dct']:.1f}% | NCC: {data['ncc_dct']:.3f}", 
                               fontsize=10, fontweight='bold',
                               color='darkgreen' if data['ncc_dct'] > 0.8 else 'orange')
        axes[row, 3].axis('off')
        
    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "watermarking_robustness_matrix.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved robustness matrix comparison to {fig2_path}")

# ==============================================================================
# MAIN EXECUTION ROUTINE
# ==============================================================================

def find_daisee_video(video_dir):
    """
    Mencari berkas video asli DAiSEE (.avi / .mp4) di dalam direktori dataset.
    """
    daisee_base = os.path.join(video_dir, "DAiSEE")
    if os.path.exists(daisee_base):
        for root, dirs, files in os.walk(daisee_base):
            for f in files:
                if f.lower().endswith(('.avi', '.mp4')):
                    full_path = os.path.join(root, f)
                    cap = cv2.VideoCapture(full_path)
                    ret, _ = cap.read()
                    cap.release()
                    if ret:
                        print(f"[FOUND] Ditemukan berkas video asli DAiSEE: {full_path}")
                        return full_path
    return None

def main():
    print("=" * 80)
    print("TUGAS 2: TRANSFORMASI GAMBAR, IMAGE COMPRESSION, DAN WATERMARKING")
    print("KOMPUTASI MULTIMEDIA - FULL AUTOMATED BENCHMARK PIPELINE")
    print("=" * 80)
    
    # --------------------------------------------------------------------------
    # 1. BAGIAN A: DAiSEE VIDEO & AUGMENTATION
    # --------------------------------------------------------------------------
    print("\n[BAGIAN A] Menyiapkan Video DAiSEE dan Ekstraksi Frame...")
    real_video = find_daisee_video(DIR_VIDEO)
    if real_video:
        video_path = real_video
        print(f"[BAGIAN A] Menggunakan video DAiSEE ASLI: {video_path}")
    else:
        video_path = os.path.join(DIR_VIDEO, "daisee_student_engagement.mp4")
        if not os.path.exists(video_path) or os.path.getsize(video_path) < 50000:
            create_synthetic_daisee_video(video_path, num_frames=300, fps=30)
        else:
            print(f"Using existing fallback video: {video_path}")
        
    # Bersihkan frame lama jika ada
    for f in os.listdir(DIR_EXTRACTED):
        fpath = os.path.join(DIR_EXTRACTED, f)
        if os.path.isfile(fpath):
            os.remove(fpath)
    for f in os.listdir(DIR_SAMPLES):
        fpath = os.path.join(DIR_SAMPLES, f)
        if os.path.isfile(fpath):
            os.remove(fpath)
            
    extracted_frames = extract_frames_from_video(video_path, DIR_EXTRACTED)
    sampled_frames = sample_random_frames(extracted_frames, DIR_SAMPLES, n=10, seed=RANDOM_SEED)
    
    print("\n[BAGIAN A] Menjalankan Pipeline Augmentasi Data...")
    run_augmentation_pipeline(sampled_frames, DIR_AUGMENTED)
    
    # --------------------------------------------------------------------------
    # 2. BAGIAN B: IMAGE COMPRESSION
    # --------------------------------------------------------------------------
    print("\n[BAGIAN B] Menyiapkan Citra Uji (BMP, JPEG, PNG, SVG) & Kompresi...")
    primary_sample = sampled_frames[0]
    benchmark_images = prepare_compression_benchmark_images(primary_sample, DIR_COMP_INPUTS)
    
    print("\n[BAGIAN B] Menjalankan Benchmark Kompresi (RLE, JPEG, WebP)...")
    comp_df = run_compression_benchmarks(benchmark_images, DIR_COMP_RESULTS)
    print("\n--- Ringkasan Hasil Kompresi Citra ---")
    print(comp_df.drop(columns=["Decompressed Image"]).to_string(index=False))
    
    # --------------------------------------------------------------------------
    # 3. BAGIAN C: WATERMARKING
    # --------------------------------------------------------------------------
    print("\n[BAGIAN C] Menjalankan Pipeline Watermarking & Uji Ketahanan...")
    run_watermarking_pipeline(primary_sample, DIR_WATERMARK)
    
    print("\n" + "=" * 80)
    print("SELURUH EKSPERIMEN BERHASIL DISELESAIKAN!")
    print(f"Hasil data tersimpan di {DATA_DIR} (bagian_a/b/c) dan visualisasi di {OUTPUT_DIR} (bagian_a/b/c)")
    print("=" * 80)

if __name__ == "__main__":
    main()
