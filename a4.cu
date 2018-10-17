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

__global__ void make_image(float* image, int size, float xSource, float ySource,
                           float* xlens, float* ylens, float rsrc, float ldc) {

}

__device__ void device_shoot(float& xSource, float& ySource, float xl, float yl,
                             float* xlens, float* ylens, float* eps, int nlenses) {

}

int main(int argc, char **argv) {
    debug("Starting Main");

    // Setting values for the lens setting 
    float *xlens, *ylens, *eps;
    const int nlenses = set_example_3(&xlens, &ylens, &eps);
    cout << "Simulating " << nlenses << " lens system." << endl;
    
    // Set the size of the lens image
    const float lens_scale = 0.005;
    const int npixx = static_cast<int>(floor((XL2 - XL1) / lens_scale)) + 1;
    const int npixy = static_cast<int>(floor((YL2 - YL1) / lens_scale)) + 1;
    cout << "Building an " << npixx << "x" << npixy << " lens image." << endl;

    Array<float, 2> lens_image(npixy, npixx);    // The lens image
    int size = lens_image.getSize();
    float* device_image = lens_image.getBuffer();

    // Source star parameters. You can adjust these if you like - it is
    // interesting to look at the different lens images that result
    const float rsrc = 0.1;      // radius
    const float ldc  = 0.5;      // limb darkening coefficient
    const float xSource = 0.0;   // x and y centre on the map
    const float ySource = 0.0;





    // Copy image to device
    cudaMalloc(&device_image, size);
    cudaMemcpy(device_image, device_image, size, cudaMemcpyHostToDevice);

    

    






    debug("Finished");
}