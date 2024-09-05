#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

#define N 3  // Dimensione del kernel (3x3)
#define BLOCK_SIZE 16  // Dimensione del blocco

__device__ inline int custom_min(int a, int b) { return a < b ? a : b; }
__device__ inline int custom_max(int a, int b) { return a > b ? a : b; }

__global__ void convoluzione2D(float* __restrict__ immagine, float* __restrict__ output, float* __restrict__ kernel, int larghezza, int altezza) {
    extern __shared__ float shared_mem[];
    float* shared_image = shared_mem;
    float* shared_kernel = shared_mem + (BLOCK_SIZE + 2) * (BLOCK_SIZE + 2);

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int kernel_radius = N / 2;

    if (tx < N && ty < N) {
        shared_kernel[ty * N + tx] = kernel[ty * N + tx];
    }

    __syncthreads();

    int xk = custom_min(custom_max(x - kernel_radius, 0), larghezza - 1);
    int yk = custom_min(custom_max(y - kernel_radius, 0), altezza - 1);

    shared_image[(ty + kernel_radius) * (BLOCK_SIZE + 2) + (tx + kernel_radius)] = immagine[yk * larghezza + xk];

    if (tx < kernel_radius) {
        shared_image[ty * (BLOCK_SIZE + 2) + tx] = immagine[yk * larghezza + custom_max(xk - kernel_radius, 0)];
    }
    if (ty < kernel_radius) {
        shared_image[ty * (BLOCK_SIZE + 2) + tx + kernel_radius] = immagine[custom_max(yk - kernel_radius, 0) * larghezza + xk];
    }
    if (tx >= blockDim.x - kernel_radius) {
        shared_image[ty * (BLOCK_SIZE + 2) + tx + 2 * kernel_radius] = immagine[yk * larghezza + custom_min(xk + kernel_radius, larghezza - 1)];
    }
    if (ty >= blockDim.y - kernel_radius) {
        shared_image[(ty + 2 * kernel_radius) * (BLOCK_SIZE + 2) + tx + kernel_radius] = immagine[custom_min(yk + kernel_radius, altezza - 1) * larghezza + xk];
    }

    __syncthreads();

    if (x >= larghezza || y >= altezza) return;

    float valore = 0.0;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            valore += shared_image[(ty + i) * (BLOCK_SIZE + 2) + (tx + j)] * shared_kernel[i * N + j];
        }
    }

    output[y * larghezza + x] = valore;
}

void scegliKernel(float kernel[N * N]) {
    int scelta;
    std::cout << "Scegli un kernel:\n";
    std::cout << "1. Identity\n";
    std::cout << "2. Edge Detection\n";
    std::cout << "3. Sharpen\n";
    std::cout << "4. Box Blur\n";
    std::cout << "5. Gaussian Blur\n";
    std::cin >> scelta;

    switch (scelta) {
    case 1:
        kernel[0] = 0; kernel[1] = 0; kernel[2] = 0;
        kernel[3] = 0; kernel[4] = 1; kernel[5] = 0;
        kernel[6] = 0; kernel[7] = 0; kernel[8] = 0;
        break;
    case 2:
        kernel[0] = 0; kernel[1] = -1; kernel[2] = 0;
        kernel[3] = -1; kernel[4] = 4; kernel[5] = -1;
        kernel[6] = 0; kernel[7] = -1; kernel[8] = 0;
        break;
    case 3:
        kernel[0] = 0; kernel[1] = -1; kernel[2] = 0;
        kernel[3] = -1; kernel[4] = 5; kernel[5] = -1;
        kernel[6] = 0; kernel[7] = -1; kernel[8] = 0;
        break;
    case 4:
        kernel[0] = 1.0 / 9; kernel[1] = 1.0 / 9; kernel[2] = 1.0 / 9;
        kernel[3] = 1.0 / 9; kernel[4] = 1.0 / 9; kernel[5] = 1.0 / 9;
        kernel[6] = 1.0 / 9; kernel[7] = 1.0 / 9; kernel[8] = 1.0 / 9;
        break;
    case 5:
        kernel[0] = 1.0 / 16; kernel[1] = 2.0 / 16; kernel[2] = 1.0 / 16;
        kernel[3] = 2.0 / 16; kernel[4] = 4.0 / 16; kernel[5] = 2.0 / 16;
        kernel[6] = 1.0 / 16; kernel[7] = 2.0 / 16; kernel[8] = 1.0 / 16;
        break;
    default:
        std::cerr << "Scelta non valida!" << std::endl;
        exit(1);
    }
}

int main() {
    cv::Mat img = cv::imread("C:/Users/imnot/Documents/UNIFI/Parallel Computing/images.jpg", cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Errore nel caricamento dell'immagine!" << std::endl;
        return -1;
    }

    int larghezza = img.cols;
    int altezza = img.rows;

    std::vector<cv::Mat> channels(3);
    cv::split(img, channels);

    float kernel[N * N];
    scegliKernel(kernel);

    float* d_kernel;
    cudaMalloc(&d_kernel, N * N * sizeof(float));
    cudaMemcpy(d_kernel, kernel, N * N * sizeof(float), cudaMemcpyHostToDevice);

    cv::Mat output_channels[3];
    for (int c = 0; c < 3; ++c) {
        channels[c].convertTo(channels[c], CV_32F);

        float* d_immagine, * d_output;
        cudaMalloc(&d_immagine, larghezza * altezza * sizeof(float));
        cudaMalloc(&d_output, larghezza * altezza * sizeof(float));

        cudaMemcpy(d_immagine, channels[c].ptr<float>(), larghezza * altezza * sizeof(float), cudaMemcpyHostToDevice);

        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize((larghezza + blockSize.x - 1) / blockSize.x, (altezza + blockSize.y - 1) / blockSize.y);

        size_t sharedMemSize = (BLOCK_SIZE + 2) * (BLOCK_SIZE + 2) * sizeof(float) + N * N * sizeof(float);

        convoluzione2D << <gridSize, blockSize, sharedMemSize >> > (d_immagine, d_output, d_kernel, larghezza, altezza);

        output_channels[c] = cv::Mat(altezza, larghezza, CV_32F);
        cudaMemcpy(output_channels[c].ptr<float>(), d_output, larghezza * altezza * sizeof(float), cudaMemcpyDeviceToHost);

        output_channels[c].convertTo(output_channels[c], CV_8U);

        cudaFree(d_immagine);
        cudaFree(d_output);
    }

    cv::Mat output_image;
    cv::merge(output_channels, 3, output_image);

    cv::imwrite("output_image.jpg", output_image);

    cudaFree(d_kernel);

    return 0;
}
