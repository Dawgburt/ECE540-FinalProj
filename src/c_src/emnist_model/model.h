#ifndef MODEL_H
#define MODEL_H

#include "weights.h"

/* Run inference on one 28x28 image. 
   Input is row-major floats in [0,1]. 
   Output is a length-47 array of probabilities. */
void predict(const float input[IMG_H][IMG_W],
             float       output[DENSE2_OUT]);

#endif
