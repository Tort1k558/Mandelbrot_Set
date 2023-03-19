#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void mandelbrotKernel(int width, int height, double minX, double maxX, double minY, double maxY, int currentIterations, uchar4 *d_output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        double xSpot = ((double)x / (double)width) * (maxX - minX) + minX;
        double ySpot = ((double)y / (double)height) * (maxY - minY) + minY;
        int n = 0;
        double real = 0;
        double imag = 0;
        while (n < currentIterations && real * real + imag * imag < 4.0) {
            double tempReal = real * real - imag * imag + xSpot;
            imag = 2 * real * imag + ySpot;
            real = tempReal;
            n++;
        }
        double t = (double)n / (double)currentIterations;
        int r = (int)(9*(1-t)*t*t*t*255);
        int g = (int)(15*(1-t)*(1-t)*t*t*255);
        int b = (int)(8.5*(1-t)*(1-t)*(1-t)*t*255);
        d_output[y * width + x] = make_uchar4(r, g, b, 255);
    }
}

void mandelbrotCuda(int width, int height, double minX, double maxX, double minY, double maxY, int currentIterations, uchar4 *d_output) {
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    mandelbrotKernel<<<gridDim, blockDim>>>(width, height, minX, maxX, minY, maxY, currentIterations, d_output);
}
