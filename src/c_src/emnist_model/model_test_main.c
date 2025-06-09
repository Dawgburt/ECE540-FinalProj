#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "model.h"    // provides: IMG_W, IMG_H, DENSE2_OUT, predict(...)
#include "weights.h"  // your generated weight arrays

//===========================================================================
// Nearest-neighbor resize for a single-channel image
//===========================================================================
static unsigned char* resize_nn(
    const unsigned char* in, int old_w, int old_h,
    int new_w, int new_h
) {
    unsigned char* out = malloc(new_w * new_h);
    if (!out) {
        fprintf(stderr, "malloc failed in resize_nn\n");
        exit(1);
    }
    for (int y = 0; y < new_h; ++y) {
        int src_y = y * old_h / new_h;
        for (int x = 0; x < new_w; ++x) {
            int src_x = x * old_w / new_w;
            out[y * new_w + x] = in[src_y * old_w + src_x];
        }
    }
    return out;
}

int main(int argc, char **argv) {
    const char* filepath = (argc > 1)
        ? argv[1]
        : "src/python_src/test_imgs/y.jpg";

    // Load as 1-channel (grayscale)
    int w, h, channels;
    unsigned char* data = stbi_load(filepath, &w, &h, &channels, 1);
    if (!data) {
        fprintf(stderr, "Error loading '%s': %s\n",
                filepath, stbi_failure_reason());
        return 1;
    }

    // Resize to 28×28 if needed
    if (w != IMG_W || h != IMG_H) {
        unsigned char* tmp = resize_nn(data, w, h, IMG_W, IMG_H);
        stbi_image_free(data);
        data = tmp;
        w = IMG_W;  h = IMG_H;
    }

    // Copy into array with EMNIST normalization:
    //    (pixel/255.0 – mean) / std
    const float EMNIST_MEAN = 0.1307f;
    const float EMNIST_STD  = 0.3081f;
    static float img[IMG_H][IMG_W];
    for (int y = 0; y < IMG_H; ++y) {
        for (int x = 0; x < IMG_W; ++x) {
            float p = data[y * IMG_W + x] / 255.0f;      // [0,1]
            img[y][x] = (p - EMNIST_MEAN) / EMNIST_STD;  // normalized
        }
    }
    free(data);

    // EMNIST data comes transposed+flipped; undo that:
    static float tmpf[IMG_H][IMG_W];
    // transpose
    for (int y = 0; y < IMG_H; ++y)
        for (int x = 0; x < IMG_W; ++x)
            tmpf[y][x] = img[x][y];
    // flip each row horizontally
    for (int y = 0; y < IMG_H; ++y)
        for (int x = 0; x < IMG_W; ++x)
            img[y][x] = tmpf[y][IMG_W - 1 - x];

    float logits[DENSE2_OUT];
    predict(img, logits);

    //  Numerically-stable softmax
    float max_logit = logits[0];
    for (int i = 1; i < DENSE2_OUT; ++i)
        if (logits[i] > max_logit)
            max_logit = logits[i];

    float sum_exp = 0.0f;
    static float probs[DENSE2_OUT];
    for (int i = 0; i < DENSE2_OUT; ++i) {
        float e = expf(logits[i] - max_logit);
        probs[i] = e;
        sum_exp += e;
    }
    for (int i = 0; i < DENSE2_OUT; ++i)
        probs[i] /= sum_exp;

    // Pick the top class
    int best = 0;
    for (int i = 1; i < DENSE2_OUT; ++i) {
        if (probs[i] > probs[best]) {
            best = i;
        }
    }

    printf("Predicted class index: %d  (probability %.4f)\n",
           best, probs[best]);
    return 0;
}
