#include "model.h"
#include <math.h>
#include <stdlib.h>

// Convolution with same padding, stride=1
static void conv2d(const float *in, int H, int W, int IC,
                   const float *kernel, int K, int OC,
                   const float *bias, float *out) {
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
                v = in[(ic*H + yy)*W + xx];
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
  int H = IMG_H, W = IMG_W;
  float *buf0 = malloc(CONV1_IN*H*W*sizeof(float));
  float *buf1 = malloc(CONV1_OUT*H*W*sizeof(float));
  float *buf2 = malloc(CONV2_OUT*H*W*sizeof(float));
  /* After pool1 dims become H/2 x W/2 */
  float *buf3 = malloc(CONV2_OUT*(H/2)*(W/2)*sizeof(float));
  float *buf4 = malloc(CONV3_OUT*(H/2)*(W/2)*sizeof(float));
  /* After pool2 dims become H/4 x W/4 */
  float *buf5 = malloc(CONV4_OUT*(H/4)*(W/4)*sizeof(float));
  float *buf6 = malloc(CONV4_OUT*(H/8)*(W/8)*sizeof(float));
  float *flat = malloc(DENSE1_IN*sizeof(float));
  float *fc1  = malloc(DENSE1_OUT*sizeof(float));

  // Copy input image into buf0 channel 0
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      buf0[y*W + x] = input[y][x];

  // ------ Block 1: Conv1 -> BN1 -> ReLU -> MaxPool ------ 
  conv2d(buf0, H, W, CONV1_IN,
         conv1_kernel, CONV_K, CONV1_OUT,
         conv1_bias, buf1);
  batchnorm(buf1, CONV1_OUT*H*W,
            bn1_gamma, bn1_beta, bn1_mean, bn1_var);
  relu(buf1, CONV1_OUT*H*W);
  maxpool2d(buf1, H, W, CONV1_OUT, buf2);

  // Now buf2 holds CONV1_OUT channels at H/2 x W/2
  H /= 2; W /= 2;

  // ------ Block 2: Conv2 -> BN2 -> ReLU -> MaxPool ------ 
  conv2d(buf2, H, W, CONV2_IN,
         conv2_kernel, CONV_K, CONV2_OUT,
         conv2_bias, buf3);
  batchnorm(buf3, CONV2_OUT*H*W,
            bn2_gamma, bn2_beta, bn2_mean, bn2_var);
  relu(buf3, CONV2_OUT*H*W);
  maxpool2d(buf3, H, W, CONV2_OUT, buf4);

  H /= 2; W /= 2;

  // ------ Block 3: Conv3 -> BN3 -> ReLU -> MaxPool ------ 
  conv2d(buf4, H, W, CONV3_IN,
         conv3_kernel, CONV_K, CONV3_OUT,
         conv3_bias, buf5);
  batchnorm(buf5, CONV3_OUT*H*W,
            bn3_gamma, bn3_beta, bn3_mean, bn3_var);
  relu(buf5, CONV3_OUT*H*W);
  maxpool2d(buf5, H, W, CONV3_OUT, buf6);

  H /= 2; W /= 2;

  // ------ Block 4: Conv4 -> BN4 -> ReLU ------
  conv2d(buf6, H, W, CONV4_IN,
         conv4_kernel, CONV_K, CONV4_OUT,
         conv4_bias, buf6);
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
