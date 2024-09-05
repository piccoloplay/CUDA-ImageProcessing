#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

#define N 3  // Dimensione del kernel (3x3)

// Funzioni min e max custom per CUDA
__device__ inline int custom_min_cuda(int a, int b) { return a < b ? a : b; }
__device__ inline int custom_max_cuda(int a, int b) { return a > b ? a : b; }

// Kernel CUDA per la convoluzione
__global__ void convoluzione2D_cuda(float* immagine, float* output, float* kernel, int larghezza, int altezza) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= larghezza || y >= altezza) return;

    float valore = 0.0;
    int kernel_radius = N / 2;

    for (int i = -kernel_radius; i <= kernel_radius; i++) {
        for (int j = -kernel_radius; j <= kernel_radius; j++) {
            int xk = custom_min_cuda(custom_max_cuda(x + i, 0), larghezza - 1);
            int yk = custom_min_cuda(custom_max_cuda(y + j, 0), altezza - 1);
            valore += immagine[yk * larghezza + xk] * kernel[(i + kernel_radius) * N + (j + kernel_radius)];
        }
    }

    output[y * larghezza + x] = valore;
}

void scegliKernelCuda(float kernel[N * N]) {
    int scelta;
    std::cout << "Scegli un kernel:\n";
    std::cout << "1. Identity\n";
    std::cout << "2. Edge Detection\n";
    std::cout << "3. Sharpen\n";
    std::cout << "4. Box Blur\n";
    std::cout << "5. Gaussian Blur\n";
    std::cin >> scelta;

    switch (scelta) {
    case 1: // Identity
        kernel[0] = 0; kernel[1] = 0; kernel[2] = 0;
        kernel[3] = 0; kernel[4] = 1; kernel[5] = 0;
        kernel[6] = 0; kernel[7] = 0; kernel[8] = 0;
        break;
    case 2: // Edge Detection
        kernel[0] = 0; kernel[1] = -1; kernel[2] = 0;
        kernel[3] = -1; kernel[4] = 4; kernel[5] = -1;
        kernel[6] = 0; kernel[7] = -1; kernel[8] = 0;
        break;
    case 3: // Sharpen
        kernel[0] = 0; kernel[1] = -1; kernel[2] = 0;
        kernel[3] = -1; kernel[4] = 5; kernel[5] = -1;
        kernel[6] = 0; kernel[7] = -1; kernel[8] = 0;
        break;
    case 4: // Box Blur
        for (int i = 0; i < N * N; i++) kernel[i] = 1.0 / 9.0;
        break;
    case 5: // Gaussian Blur
        kernel[0] = 1; kernel[1] = 2; kernel[2] = 1;
        kernel[3] = 2; kernel[4] = 4; kernel[5] = 2;
        kernel[6] = 1; kernel[7] = 2; kernel[8] = 1;
        for (int i = 0; i < N * N; i++) kernel[i] /= 16.0;
        break;
    default:
        std::cerr << "Scelta non valida!\n";
        exit(-1);
    }
}

int mainZ() {
    // Carica l'immagine a colori usando OpenCV
    cv::Mat img = cv::imread("C:/Users/imnot/Documents/UNIFI/Parallel Computing/images.jpg", cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Errore nel caricamento dell'immagine!" << std::endl;
        return -1;
    }

    int larghezza = img.cols;
    int altezza = img.rows;

    // Alloca memoria per i canali separati (rosso, verde, blu)
    std::vector<cv::Mat> channels(3);
    cv::split(img, channels);

    float kernel[N * N];
    scegliKernelCuda(kernel);

    float* d_kernel;
    cudaMalloc(&d_kernel, N * N * sizeof(float));
    cudaMemcpy(d_kernel, kernel, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Eventi per misurare il tempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Inizio della misurazione del tempo
    cudaEventRecord(start);

    // Convolvi ogni canale separatamente
    cv::Mat output_channels[3];
    for (int c = 0; c < 3; ++c) {
        // Conversione del canale in float per GPU
        channels[c].convertTo(channels[c], CV_32F);

        float* d_immagine, * d_output;
        cudaMalloc(&d_immagine, larghezza * altezza * sizeof(float));
        cudaMalloc(&d_output, larghezza * altezza * sizeof(float));

        // Copia il canale dell'immagine in memoria GPU
        cudaMemcpy(d_immagine, channels[c].ptr<float>(), larghezza * altezza * sizeof(float), cudaMemcpyHostToDevice);

        // Configura il lancio del kernel
        dim3 blockSize(16, 16);
        dim3 gridSize((larghezza + blockSize.x - 1) / blockSize.x, (altezza + blockSize.y - 1) / blockSize.y);
        convoluzione2D_cuda << <gridSize, blockSize >> > (d_immagine, d_output, d_kernel, larghezza, altezza);

        // Copia il risultato dal canale convoluto dalla GPU alla CPU
        output_channels[c] = cv::Mat(altezza, larghezza, CV_32F);
        cudaMemcpy(output_channels[c].ptr<float>(), d_output, larghezza * altezza * sizeof(float), cudaMemcpyDeviceToHost);

        // Riconversione in uint8 per visualizzare e salvare correttamente l'immagine
        output_channels[c].convertTo(output_channels[c], CV_8U);

        cudaFree(d_immagine);
        cudaFree(d_output);
    }

    // Fine della misurazione del tempo
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Combina i canali in un'unica immagine
    cv::Mat output_img;
    cv::merge(output_channels, 3, output_img);

    // Salva l'immagine convoluta
    cv::imwrite("output.jpg", output_img);

    // Calcola e stampa il tempo di esecuzione del kernel
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Tempo di esecuzione del kernel: " << milliseconds << " ms" << std::endl;

    // Pulizia
    cudaFree(d_kernel);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
