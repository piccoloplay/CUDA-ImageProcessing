# CUDA Image Processing Project

This project implements a 2D image convolution using CUDA for parallel image processing. The main goal is to leverage the GPU's parallelization capabilities to achieve significant performance improvements compared to a sequential CPU implementation. OpenCV is used for image handling.
[![Watch the video](https://img.youtube.com/vi/jjCtI-Ygrhg/maxresdefault.jpg)](https://www.youtube.com/watch?v=jjCtI-Ygrhg)

## Table of Contents
- [Introduction](#introduction)
- [Algorithm](#algorithm)
- [Technical Implementation](#technical-implementation)
- [Benchmark and Results](#benchmark-and-results)
- [License](#license)

## Introduction
The project focuses on implementing a 2D convolution algorithm using CUDA to process images. By using the GPU's parallel computation capabilities, the performance of the convolution is enhanced compared to a traditional CPU-based approach. OpenCV is used for handling image input/output and channel conversion, while CUDA is employed for the parallel computation of the convolution.

## Algorithm
The algorithm applies a user-defined kernel to perform a 2D convolution on an image. Each channel of the image (Red, Green, Blue) is processed separately in parallel on the GPU, using blocks and threads to process the pixels concurrently.

### Key Points of the Algorithm:
- For each pixel in the image, the kernel is applied over a 3x3 area around the pixel.
- The image edges are handled using clamping of the index values (via custom min/max functions).
- Shared memory is used to minimize the number of accesses to global memory.

## Technical Implementation

### Libraries Used
- **CUDA**: Used for performing the parallel convolution.
- **OpenCV**: Used for image handling, such as loading, saving, and channel conversion.

### Parallelization with CUDA
The CUDA code utilizes a grid of blocks and threads to parallelize the application of the convolution kernel. Each thread processes a single pixel in the image, applying the kernel to that pixel. Shared memory is used to store the portion of the image and the kernel, reducing global memory accesses and improving performance.

## Benchmark and Results

The performance test was conducted using a fixed kernel of size 3x3. The focus was on comparing a simple version (without specific optimizations) with an optimized version using shared memory.

### Execution Times
| Version                            | Execution Time (ms) |
|-------------------------------------|---------------------|
| Simple Version                      | 18.68 ms            |
| Optimized Version (with Shared Memory) | 0.098 ms            |

### Performance Explanation
The optimized version achieves a significant performance boost, primarily due to:
1. **Use of Shared Memory**: By utilizing shared memory to store the kernel data and a portion of the image, the number of accesses to global memory is drastically reduced. Shared memory is much faster than global memory, and this optimization allows threads to access data more efficiently and in parallel.
2. **Thread Synchronization**: The use of `__syncthreads()` ensures that all necessary data is loaded into shared memory before computations begin. This avoids race conditions and ensures that each thread has access to the correct data, improving the accuracy and efficiency of parallel execution.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
