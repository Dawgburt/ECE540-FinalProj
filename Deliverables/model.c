#include "model.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

// Convolution with same padding, stride=1
static void conv2d(const float *in, int H, int W, int IC,
                   const float *kernel, int K, int OC,
                   const float *bias, float *out, 
                   const char *layer_name) {
  int pad = K/2;
  for (int oc = 0; oc < OC; ++oc) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        float sum = 0.0f;
        for (int ic = 0; ic < IC; ++ic) {
          for (int ky = 0; ky < K; ++ky) {
            for (int kx = 0; kx < K; ++kx) {
              int yy = y + ky - pad;
              int xx = x + kx - pad;
              float v = 0.0f;
              if (yy >= 0 && yy < H && xx >= 0 && xx < W) {
                size_t idx_in = (ic*H + yy)*W + xx;
                if (isnan(in[idx_in])) {
                  printf(
                    "%s: NaN in in[%zu] at ic=%d y=%d x=%d    (yy=%d, xx=%d)\n",
                    layer_name,    // ← %s
                    idx_in,        // ← %zu
                    ic, y, x, yy, xx
                  );
                  break;
                }
                v = in[idx_in];
              }
              int idx = ((oc*IC + ic)*K + ky)*K + kx;
              sum += v * kernel[idx];
            }
          }
        }
        out[(oc*H + y)*W + x] = sum + bias[oc];
      }
    }
  }
}

// BatchNorm inference 
static void batchnorm(float *data, int N,
                      const float *gamma, const float *beta,
                      const float *mean,  const float *var) {
  const float eps = 1e-5f;
  for (int i = 0; i < N; ++i) {
    data[i] = gamma[i] * (data[i] - mean[i]) / sqrtf(var[i] + eps)
              + beta[i];
  }
}

// ReLU activation function
static void relu(float *data, int N) {
  for (int i = 0; i < N; ++i) {
    if (data[i] < 0) data[i] = 0;
  }
}

// 2x2 max-pool stride 2
static void maxpool2d(const float *in, int H, int W, int C,
                      float *out) {
  int HO = H/2, WO = W/2;
  for (int c = 0; c < C; ++c) {
    for (int y = 0; y < HO; ++y) {
      for (int x = 0; x < WO; ++x) {
        float m = in[(c*H + (2*y)) * W + (2*x)];
        for (int dy = 0; dy < 2; ++dy) {
          for (int dx = 0; dx < 2; ++dx) {
            float v = in[(c*H + 2*y+dy)*W + (2*x+dx)];
            if (v > m) m = v;
          }
        }
        out[(c*HO + y)*WO + x] = m;
      }
    }
  }
}

// Fully-connected
static void dense(const float *in, int NI,
                  const float *kernel, int NO,
                  const float *bias, float *out) {
  for (int o = 0; o < NO; ++o) {
    float sum = 0.0f;
    for (int i = 0; i < NI; ++i) {
      sum += in[i] * kernel[o*NI + i];
    }
    out[o] = sum + bias[o];
  }
}

// Softmax in-place
static void softmax(float *v, int N) {
  float m = v[0];
  for (int i = 1; i < N; ++i) if (v[i] > m) m = v[i];
  float s = 0.0f;
  for (int i = 0; i < N; ++i) {
    v[i] = expf(v[i] - m);
    s   += v[i];
  }
  for (int i = 0; i < N; ++i) {
    v[i] /= s;
  }
}

// The full forward pass
void predict(const float input[IMG_H][IMG_W],
             float       output[DENSE2_OUT])
{
  // Buffers for each stage
int H  = IMG_H;
int W  = IMG_W;


int   C1 = CONV1_OUT;          // number of channels after Conv1
float *buf0 = calloc(CONV1_IN  * H * W, sizeof(float));  // input copy
float *buf1 = calloc(C1       * H * W, sizeof(float));  // Conv1 output

// 2) After Pool1 -> dims H1×W1
int H1 = H/2;
int W1 = W/2;
float *buf2 = calloc(C1 * H1 * W1, sizeof(float));       // Pool1 output

// Conv2 runs on H1×W1 -> produces H1×W1 with CONV2_OUT channels
int   C2 = CONV2_OUT;
float *buf3 = calloc(C2 * H1 * W1, sizeof(float));       // Conv2 output

// Pool2 shrinks H1×W1 -> H2×W2
int H2 = H1/2;
int W2 = W1/2;
float *buf4 = calloc(C2 * H2 * W2, sizeof(float));       // Pool2 output

// Conv3 on H2×W2 -> produces H2×W2 with CONV3_OUT channels
int   C3 = CONV3_OUT;
float *buf5 = calloc(C3 * H2 * W2, sizeof(float));       // Conv3 output

// Pool3 shrinks H2×W2 -> H3×W3
int H3 = H2/2;
int W3 = W2/2;
float *buf6 = calloc(C3 * H3 * W3, sizeof(float));       // Pool3 output

// Conv4 (no pooling) on H3×W3 -> produces H3×W3 with CONV4_OUT channels
int   C4 = CONV4_OUT;
float *buf7 = calloc(C4 * H3 * W3, sizeof(float));       // Conv4 output

// Flatten and the dense layers
int flat_size = C4 * H3 * W3;
float *flat = calloc(flat_size, sizeof(float));

float *fc1  = calloc(DENSE1_OUT, sizeof(float));         // Dense1 output
float *out  = calloc(DENSE2_OUT, sizeof(float));         // final softmax output

// Copy input image into buf0 channel 0
for (int y = 0; y < H; ++y)
  for (int x = 0; x < W; ++x)
    buf0[y*W + x] = input[y][x];


  // ------ Block 1: Conv1 -> BN1 -> ReLU -> MaxPool ------ 
  conv2d(buf0, H, W, CONV1_IN,
         conv1_kernel, CONV_K, CONV1_OUT,
         conv1_bias, buf1, "Layer 1");
  batchnorm(buf1, CONV1_OUT*H*W,
            bn1_gamma, bn1_beta, bn1_mean, bn1_var);

  float val0 = buf1[0], val1 = buf1[H*W]; 
  printf("Conv1 raw  buf1[0]=%f  buf1[H*W]=%f\n", val0, val1);
  relu(buf1, CONV1_OUT*H*W);
  val0 = buf1[0];
  val1 = buf1[H*W];
  printf("Conv1 raw  buf1[0]=%f  buf1[H*W]=%f\n", val0, val1);

  maxpool2d(buf1, H, W, CONV1_OUT, buf2);

  // buf2 holds CONV1_OUT channels at H/2 x W/2
  H /= 2; W /= 2;

  // ------ Block 2: Conv2 -> BN2 -> ReLU -> MaxPool ------ 
  conv2d(buf2, H, W, CONV2_IN,
         conv2_kernel, CONV_K, CONV2_OUT,
         conv2_bias, buf3, "Layer 1");
  batchnorm(buf3, CONV2_OUT*H*W,
            bn2_gamma, bn2_beta, bn2_mean, bn2_var);
  relu(buf3, CONV2_OUT*H*W);
  maxpool2d(buf3, H, W, CONV2_OUT, buf4);

  H /= 2; W /= 2;

  // ------ Block 3: Conv3 -> BN3 -> ReLU -> MaxPool ------ 
  conv2d(buf4, H, W, CONV3_IN,
         conv3_kernel, CONV_K, CONV3_OUT,
         conv3_bias, buf5, "Layer 3");
  batchnorm(buf5, CONV3_OUT*H*W,
            bn3_gamma, bn3_beta, bn3_mean, bn3_var);
  relu(buf5, CONV3_OUT*H*W);
  maxpool2d(buf5, H, W, CONV3_OUT, buf6);

  H /= 2; W /= 2;

  // ------ Block 4: Conv4 -> BN4 -> ReLU ------
  conv2d(buf6, H, W, CONV4_IN,
         conv4_kernel, CONV_K, CONV4_OUT,
         conv4_bias, buf6, "Layer 4");
  batchnorm(buf6, CONV4_OUT*H*W,
            bn4_gamma, bn4_beta, bn4_mean, bn4_var);
  relu(buf6, CONV4_OUT*H*W);

  // Flatten into array "flat"
  for (int c = 0; c < CONV4_OUT; ++c)
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x)
        flat[(c*H + y)*W + x] = buf6[(c*H + y)*W + x];

  // ------ Dense1 -> BN5 -> ReLU ------
  dense(flat, DENSE1_IN, dense1_kernel, DENSE1_OUT, dense1_bias, fc1);
  batchnorm(fc1, DENSE1_OUT,
            bn5_gamma, bn5_beta, bn5_mean, bn5_var);
  relu(fc1, DENSE1_OUT);

  // ------ Dense2 -> Softmax ------
  dense(fc1, DENSE1_OUT, dense2_kernel, DENSE2_OUT, dense2_bias, output);
  softmax(output, DENSE2_OUT);

  // free buffers
  free(buf0); free(buf1); free(buf2);
  free(buf3); free(buf4); free(buf5);
  free(buf6); free(flat); free(fc1);
}
