#!/usr/bin/env python3
"""
test_img_to_bin.py

Converts a 28×28 JPEG (or any size) to raw binary FP32 matrix for EMNIST C inference.

Usage:
    python test_img_to_bin.py input.jpg output.bin
"""

import sys
import numpy as np
from PIL import Image

def convert(jpeg_path, bin_path):
    # Load and convert to grayscale ('L')
    img = Image.open(jpeg_path).convert("L")
    
    # Determine appropriate resampling filter
    try:
        resample_mode = Image.Resampling.LANCZOS
    except AttributeError:
        # Older Pillow versions
        resample_mode = Image.LANCZOS

    # Resize to 28x28
    img = img.resize((28, 28), resample=resample_mode)
    
    # To numpy array (28×28 uint8)
    arr = np.array(img, dtype=np.uint8)
    
    # Transpose + flip (match TF pipeline)
    arr = arr.T
    arr = np.fliplr(arr)
    
    # Normalize to [0,1] float32
    mat = arr.astype(np.float32) / 255.0
    
    # Write row-major float32
    mat.tofile(bin_path)
    print(f"Wrote {bin_path}: {mat.size} floats")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: test_img_to_bin.py input.jpg output.bin")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
