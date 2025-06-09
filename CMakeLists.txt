cmake_minimum_required(VERSION 3.10)
project(ECE540_FINALPROJ C)

# where to put the finished binary
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

# turn on debug symbols
set(CMAKE_BUILD_TYPE Debug)

# point at your model directory
set(MODEL_DIR ${PROJECT_SOURCE_DIR}/src/c_src/emnist_model)

# our “library” source
set(MODEL_SRC 
    ${MODEL_DIR}/model.c
)

# the entry-point
set(MAIN_SRC 
    ${MODEL_DIR}/model_test_main.c
)

# headers live here
include_directories(${MODEL_DIR})

# define the executable
add_executable(model_test_main
    ${MAIN_SRC}
    ${MODEL_SRC}
)

# ← Link against libm for pow(), sqrtf(), expf(), etc.
target_link_libraries(model_test_main PRIVATE m)
