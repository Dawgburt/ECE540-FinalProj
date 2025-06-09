from PIL import Image
import numpy as np

# === Configuration ===
# Path to your 28×28 grayscale image file:
INPUT_PATH = 'test_imgs/8.jpg'
# Path to the generated header file:
OUTPUT_PATH = 'test8.h'
# C variable name for the generated array:
VAR_NAME = 'test_image'
# =====================


def generate_header(input_path, output_path, var_name):
    # Load image, convert to grayscale, resize to 28×28 using LANCZOS for high-quality downsampling
    img = Image.open(input_path).convert('L')
    img = img.resize((28, 28), Image.LANCZOS)
    # Normalize pixel values to [0,1]
    arr = np.asarray(img, dtype=np.float32) / 255.0
    flat = arr.flatten()

    # Prepare include guard based on variable name
    guard = var_name.upper() + '_H'

    # Write C header file
    with open(output_path, 'w') as f:
        f.write(f'#ifndef {guard}\n')
        f.write(f'#define {guard}\n\n')
        f.write('#include <vector>\n\n')
        f.write(f'// Generated from {input_path}\n')
        f.write(f'static const float {var_name}[784] = {{\n')
        for i, val in enumerate(flat):
            f.write(f'    {val:.6f}f')
            if i < len(flat) - 1:
                f.write(',')
            if (i + 1) % 16 == 0:
                f.write('\n')
            else:
                f.write(' ')
        f.write('\n};\n\n')
        f.write(f'inline std::vector<float> get_{var_name}() {{\n')
        f.write(f'    return std::vector<float>({var_name}, {var_name} + 784);\n')
        f.write('}\n\n')
        f.write(f'#endif // {guard}\n')

if __name__ == '__main__':
    generate_header(INPUT_PATH, OUTPUT_PATH, VAR_NAME)
