#include <iostream>
#include <stdio.h>
#include <cstdarg>

#include "lenses.h"
#include "arrayff.hxx"

#define DEBUG

// TODO: Get the make_image parameters going and start writing that function. 

using namespace std;

const float WL  = 2.0;
const float XL1 = -WL;
const float XL2 =  WL;
const float YL1 = -WL;
const float YL2 =  WL;

void debug(const char* format, ...) {
    #ifdef DEBUG
    va_list args;
    va_start(args, format);
    vfprintf(stdout, "DEBUG :: ", NULL);
    vfprintf(stdout, format, args);
    vfprintf(stdout, "\n", NULL);
    va_end(args);
    fflush(stdout);
    #endif
}

void print(const char* format, ...) {
    va_list args;
    va_start(args, format);
    // vfprintf(stdout, "", NULL);
    vfprintf(stdout, format, args);
    vfprintf(stdout, "\n", NULL);
    va_end(args);
    fflush(stdout);
}

__device__ int getY(int id, int xdim, int ydim) {
    return (int) floorf(id / xdim);
}

__device__ int getX(int id, int xdim, int ydim) {
    return id % xdim;
}

__device__ void device_shoot(float& xSource, float& ySource, float xl, float yl,
                             float* xlens, float* ylens, float* eps, int nlenses) {
    float dx, dy, dr;
    xSource = xl;
    ySource = yl;
    for (int p = 0; p < nlenses; ++p) {
        dx = xl - xlens[p];
        dy = yl - ylens[p];
        dr = dx * dx + dy * dy;
        xSource -= eps[p] * dx / dr;
        ySource -= eps[p] * dy / dr;
    }
}

// the lens image 
// the size of lens image
// the parameters of the image
// lens scale, int xdim, int ydim
// lens info, with eps & nlenses
// source star info (x, y, radius^2)
// lens darkening coefficient
// Total: 17

// Ones to copy over to memory:
// image, xlens, ylens, eps

__global__ void make_image(float* image, int size,                                  // lens image
                           float XLeft, float XRight, float YBottom, float YTop,    // the size of lens image
                           float lens_scale, int xDim, int yDim,                    // lens scale, x dim, ydim
                           float* xlens, float* ylens, float* eps, int num_lenses,  // lens info with eps and number of lenses
                           float xSource, float ySource, float radSource2,          // source star information, x, y, radius^2
                           float ldc) {                                             // lens darkening coefficient
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    int ix = getX(id, xDim, yDim);
    int iy = getY(id, xDim, yDim);

    // int xl = XLeft + ix * lens_scale;
    // int yl = YBottom + iy * lens_scale;

    // float xs, ys;

    // device_shoot(xs, ys, xl, yl, xlens, ylens, eps, num_lenses);

    // int xd = xs - xSource;
    // int yd = ys - ySource;
    // int sep2 = xd * xd + yd * yd;
    // if (sep2 < radSource2) {
    //     float mu = sqrt(1 - (sep2 / radSource2));
    //     image[id] = 1.0 - ldc * (1 - mu);
    // }

    image[id] = id;
    image[id] = 100;
    
}

void checkForError(cudaError_t error) {
    if (error != cudaSuccess) {
        cout << "Error: " << cudaGetErrorString(error) << endl;
    }
}

int main(int argc, char **argv) {
    debug("Starting Main");

    // Setting values for the lens setting 
    float *xlens, *ylens, *eps;
    const int nlenses = set_example_1(&xlens, &ylens, &eps);
    print("Simulating a %d lens system", nlenses);
    
    // Set the size of the lens image
    const float lens_scale = 0.005;
    const int npixx = static_cast<int>(floor((XL2 - XL1) / lens_scale)) + 1;
    const int npixy = static_cast<int>(floor((YL2 - YL1) / lens_scale)) + 1;
    print("Building a %d by %d image", npixx, npixy);

    // Initialise image and get information
    Array<float, 2> lens_image(npixy, npixx);   
    int size = lens_image.getSize();

    // Source star parameters. You can adjust these if you like - it is
    // interesting to look at the different lens images that result
    const float rsrc = 0.1;      // radius
    const float ldc  = 0.5;      // limb darkening coefficient
    const float xSource = 0.0;   // x and y centre on the map
    const float ySource = 0.0;
    const float rsrc2 = rsrc * rsrc;

    // Create some device addresses
    float* device_image;
    float* device_xlens;
    float* device_ylens;
    float* device_eps;

    cudaDeviceSynchronize();
    printf("Cuda status: %s\n", cudaGetErrorString(cudaGetLastError()));

    // initialise memory for device
    checkForError(cudaMalloc(&device_image, size));
    checkForError(cudaMalloc(&device_xlens, nlenses));
    checkForError(cudaMalloc(&device_ylens, nlenses));
    checkForError(cudaMalloc(&device_eps, nlenses));

    cudaDeviceSynchronize();
    printf("Cuda status: %s\n", cudaGetErrorString(cudaGetLastError()));

    // Copy memory over onto device
    checkForError(cudaMemcpy(device_image, lens_image.getBuffer(), size, cudaMemcpyHostToDevice));
    checkForError(cudaMemcpy(device_xlens, xlens, nlenses, cudaMemcpyHostToDevice));
    checkForError(cudaMemcpy(device_ylens, ylens, nlenses, cudaMemcpyHostToDevice));
    checkForError(cudaMemcpy(device_eps, eps, nlenses, cudaMemcpyHostToDevice));

    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

    debug("Dimensions are tpb: %d, bpg: %d", threads_per_block, blocks_per_grid);
    print("Launching a grid of %d %d threads", blocks_per_grid, threads_per_block*blocks_per_grid);

    make_image<<<blocks_per_grid, threads_per_block>>>(device_image, size,
                                                       XL1, XL2, YL1, YL2,
                                                       lens_scale, npixx, npixy,
                                                       device_xlens, device_ylens, device_eps, nlenses,
                                                       xSource, ySource, rsrc2,
                                                       ldc);

    // Copy it back
    float test_array[size];
    print("First three:\n%f, %f, %f\nLast three:\n%f, %f, %f", test_array[0], test_array[1],
                                                               test_array[2], test_array[size-3],
                                                               test_array[size-2], test_array[size-1]);

    checkForError(cudaMemcpy(test_array, device_image, size, cudaMemcpyDeviceToHost));
    
    dump_array<float, 2>(lens_image, "cuda_lens.fit");

    print("First three:\n%f, %f, %f\nLast three:\n%f, %f, %f", test_array[0], test_array[1],
                                                               test_array[2], test_array[size-3],
                                                               test_array[size-2], test_array[size-1]);

    // Free the memory
    cudaFree(device_image);
    cudaFree(device_xlens);
    cudaFree(device_ylens);
    cudaFree(device_eps);

    debug("Finished");
}