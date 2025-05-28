"""
weights_header_generator.py

Reads the EMNIST weights archive from
    src/python_src/weight_files/emnist_weights.npz
and creates a C header at
    src/c_src/weights.h
with all parameters as `static const float` arrays.
"""

import numpy as np
from pathlib import Path

# Specify paths

# This script's folder: src/python_src
script_dir   = Path(__file__).resolve().parent

# Where  emnist_weights.npz  lives:
weights_dir  = script_dir / "weight_files"
npz_path     = weights_dir / "emnist_weights.npz"


output_header = script_dir.parent / "c_src" / "weights.h"

# Load the saved weights

if not npz_path.exists():
    raise FileNotFoundError(f"Cannot find weights archive:\n  {npz_path}")

archive = np.load(npz_path)

# Generate the C header

guard = "WEIGHTS_H"
with open(output_header, "w") as f:
    f.write(f"#ifndef {guard}\n")
    f.write(f"#define {guard}\n\n")

    for key, array in archive.items():
        # convert "layer/kernel" -> "layer_kernel"
        ident = key.replace("/", "_")
        flat  = array.astype(np.float32).ravel()
        count = flat.size

        # write declaration
        f.write(f"static const float {ident}[{count}] = {{\n")

        # create 8 values per line in scientific notation
        for i, val in enumerate(flat):
            comma = "," if i < count - 1 else ""
            f.write(f"    {val:.8e}{comma}")
            if (i + 1) % 8 == 0:
                f.write("\n")
        if count % 8:
            f.write("\n")

        f.write("};\n\n")

    f.write(f"#endif  // {guard}\n")

print(f"Generated header: {output_header}")
