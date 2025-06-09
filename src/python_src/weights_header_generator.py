"""
weights_header_generator.py

Reads the EMNIST weights archive from
    src/python_src/weight_files/emnist_weights.npz
and emits a single C header with:
  - model dimension macros
  - embedded `static const float` arrays named exactly as in model.c
into
    src/c_src/weights_embedded.h
"""

import numpy as np
from pathlib import Path

# --- Paths ---
script_dir    = Path(__file__).resolve().parent
weights_dir   = script_dir / "weight_files"
npz_path      = weights_dir / "emnist_weights.npz"
output_header = script_dir.parent / "c_src" / "emnist_model" / "weights.h"

# --- Check archive exists ---
if not npz_path.exists():
    raise FileNotFoundError(f"Cannot find weights archive:\n  {npz_path}")

archive = np.load(npz_path)

# --- Mapping from npz keys to C identifiers in model.c ---
name_map = {
    "conv2d/kernel":                    "conv1_kernel",
    "conv2d/bias":                      "conv1_bias",
    "batch_normalization/gamma":        "bn1_gamma",
    "batch_normalization/beta":         "bn1_beta",
    "batch_normalization/moving_mean":  "bn1_mean",
    "batch_normalization/moving_variance":"bn1_var",

    "conv2d_1/kernel":                  "conv2_kernel",
    "conv2d_1/bias":                    "conv2_bias",
    "batch_normalization_1/gamma":      "bn2_gamma",
    "batch_normalization_1/beta":       "bn2_beta",
    "batch_normalization_1/moving_mean":"bn2_mean",
    "batch_normalization_1/moving_variance":"bn2_var",

    "conv2d_2/kernel":                  "conv3_kernel",
    "conv2d_2/bias":                    "conv3_bias",
    "batch_normalization_2/gamma":      "bn3_gamma",
    "batch_normalization_2/beta":       "bn3_beta",
    "batch_normalization_2/moving_mean":"bn3_mean",
    "batch_normalization_2/moving_variance":"bn3_var",

    "conv2d_3/kernel":                  "conv4_kernel",
    "conv2d_3/bias":                    "conv4_bias",
    "batch_normalization_3/gamma":      "bn4_gamma",
    "batch_normalization_3/beta":       "bn4_beta",
    "batch_normalization_3/moving_mean":"bn4_mean",
    "batch_normalization_3/moving_variance":"bn4_var",

    "dense/kernel":                     "dense1_kernel",
    "dense/bias":                       "dense1_bias",
    "batch_normalization_4/gamma":      "bn5_gamma",
    "batch_normalization_4/beta":       "bn5_beta",
    "batch_normalization_4/moving_mean":"bn5_mean",
    "batch_normalization_4/moving_variance":"bn5_var",

    "dense_1/kernel":                   "dense2_kernel",
    "dense_1/bias":                     "dense2_bias"
}

# --- Write the C header ---
guard = "WEIGHTS_EMBEDDED_H"
with open(output_header, "w") as f:
    # Include guards
    f.write(f"#ifndef {guard}\n")
    f.write(f"#define {guard}\n\n")

    # Model dimension macros
    f.write("""
// image dims
#define IMG_H       28
#define IMG_W       28

// conv layers
#define CONV1_IN    1
#define CONV1_OUT   32
#define CONV2_IN    CONV1_OUT
#define CONV2_OUT   32
#define CONV3_IN    CONV2_OUT
#define CONV3_OUT   64
#define CONV4_IN    CONV3_OUT
#define CONV4_OUT   64
#define CONV_K      3

// dense layers
#define DENSE1_IN   ((IMG_H/4)*(IMG_W/4)*CONV4_OUT)
#define DENSE1_OUT  256
#define DENSE2_IN   DENSE1_OUT
#define DENSE2_OUT  47

""")

    # Embedded arrays
    for key, cname in name_map.items():
        if key not in archive:
            raise KeyError(f"Missing key in npz: {key}")
        arr  = archive[key].astype(np.float32).ravel()
        size = arr.size

        # Declaration
        f.write(f"static const float {cname}[{size}] = {{\n")
        for i, v in enumerate(arr):
            comma = "," if i < size-1 else ""
            f.write(f"    {v:.8e}{comma}")
            if (i+1) % 8 == 0:
                f.write("\n")
        if size % 8:
            f.write("\n")
        f.write("};\n\n")

    f.write(f"#endif  /* {guard} */\n")
    

print(f"✔ Generated header: {output_header}")
