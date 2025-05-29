// src/c_src/main.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"           // put stb_image.h in your c_src folder

#include "model.h"               // declares predict() and IMG_H, IMG_W, DENSE2_OUT
#include "weights.h"    // your generated header with macros + arrays

// Nearest-neighbor resize for a single-channel image
static unsigned char* resize_nn(const unsigned char* in, int w, int h, int new_w, int new_h) {
    unsigned char* out = malloc(new_w * new_h);
    if (!out) {
        fprintf(stderr, "malloc failed in resize_nn\n");
        exit(1);
    }
    for (int y = 0; y < new_h; ++y) {
        int src_y = y * h / new_h;
        for (int x = 0; x < new_w; ++x) {
            int src_x = x * w / new_w;
            out[y * new_w + x] = in[src_y * w + src_x];
        }
    }
    return out;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <image_file>\n", argv[0]);
        return 1;
    }
    //const char* filepath = argv[1];

    const char* filepath = "src/python_src/test_imgs/0.jpg";

    // 1) Load image as 1-channel (grayscale)
    int w, h, channels;
    unsigned char* data = stbi_load(filepath, &w, &h, &channels, 1);
    if (!data) {
        fprintf(stderr, "Error loading '%s': %s\n", filepath, stbi_failure_reason());
        return 1;
    }

    // 2) Resize if needed
    if (w != IMG_W || h != IMG_H) {
        unsigned char* resized = resize_nn(data, w, h, IMG_W, IMG_H);
        stbi_image_free(data);
        data = resized;
        w = IMG_W;
        h = IMG_H;
    }

    // 3) Prepare float image buffer
    float img[IMG_H][IMG_W];
    // Copy & normalize
    for (int y = 0; y < IMG_H; ++y) {
        for (int x = 0; x < IMG_W; ++x) {
            img[y][x] = data[y * IMG_W + x] / 255.0f;
        }
    }
    free(data);

    // 4) Transpose + flip left-right
    float tmp[IMG_H][IMG_W];
    // Transpose
    for (int y = 0; y < IMG_H; ++y)
        for (int x = 0; x < IMG_W; ++x)
            tmp[y][x] = img[x][y];
    // Flip each row
    for (int y = 0; y < IMG_H; ++y)
        for (int x = 0; x < IMG_W; ++x)
            img[y][x] = tmp[y][IMG_W - 1 - x];

    // 5) Run inference
    float out[DENSE2_OUT];
    predict(img, out);

    // 6) Find the highest-probability class
    int best = 0;
    for (int i = 1; i < DENSE2_OUT; ++i) {
        if (out[i] > out[best]) best = i;
    }
    printf("Predicted class index: %d (probability %.4f)\n", best, out[best]);

    return 0;
}
