// Purpose:    Elementwise addition of three vectors (d = a + b + c)
//------------------------------------------------------------------------------

#define __CL_ENABLE_EXCEPTIONS

#include "cl.hpp"

#include "util.hpp" // utility library

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <iostream>
#include <fstream>

// pick up device type from compiler command line or from the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_GPU
#endif

#include "err_code.h"

//------------------------------------------------------------------------------

#define TOL    (0.001)   // tolerance used in floating point comparisons
#define LENGTH (4)    // length of vectors a, b, and c

int main(void)
{
    std::vector<float> h_a(LENGTH);                // a vector
    std::vector<float> h_b(LENGTH);                // b vector
    std::vector<float> h_c(LENGTH);                // c vector
    std::vector<float> h_d (LENGTH, 0xdeadbeef);   // d vector (result)
    // for now d is filled with hex default values 0xdeadbeef

    cl::Buffer d_a;                       // device memory used for the input  a vector
    cl::Buffer d_b;                       // device memory used for the input  b vector
    cl::Buffer d_c;                       // device memory used for the input c vector
    cl::Buffer d_d;                       // device memory used for the output d vector

    // Fill vectors a and b with random float values
    int count = LENGTH;
    for(int i = 0; i < count; i++)
    {
        h_a[i]  = rand() % 100;
        h_b[i]  = rand() % 100;
        h_c[i]  = rand() % 100;
    }

    try
    {
    	// Create a context
        cl::Context context(DEVICE);

        // Load in kernel source, creating a program object for the context

        cl::Program program(context, util::loadProgram("vadd_abc.cl"), true);

        // Get the command queue
        cl::CommandQueue queue(context);

        // Create the kernel functor
        // 4 buffers passed as arg to kernel i.e. a, b, c, d 
        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int> vadd(program, "vadd");

        d_a   = cl::Buffer(context, h_a.begin(), h_a.end(), true); // true here means readonly flag
        d_b   = cl::Buffer(context, h_b.begin(), h_b.end(), true);
        d_c   = cl::Buffer(context, h_c.begin(), h_c.end(), true);

        d_d  = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * LENGTH);

        util::Timer timer;
        vadd(
            cl::EnqueueArgs(
                queue,
                cl::NDRange(count)),
            d_a,
            d_b,
            d_c,
            d_d,
            count);

        cl::copy(queue, d_d, h_d.begin(), h_d.end());
        
        double rtime = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
        printf("\nThe kernels ran in %lf seconds\n", rtime);
        
        // Test the results
        int correct = 0;
        float tmp;
        for(int i = 0; i < count; i++)
        {
            tmp = h_a[i] + h_b[i] + h_c[i];              // assign element i of a+b+c to tmp
            printf("inputs are %f , %f , %f and \n sum is %f \n", h_a[i], h_b[i], h_c[i] , h_d[i]);
            tmp -= h_d[i];                               // compute deviation of expected and output result
            if(tmp*tmp < TOL*TOL)                        // correct if square deviation is less than tolerance squared
                correct++;
            else {
                printf(" tmp %f h_a %f h_b %f h_c %f h_d %f\n",tmp, h_a[i], h_b[i], h_c[i], h_d[i]);
            }
        }

        // summarize results
        printf("D = A+B+C:  %d out of %d results were correct.\n", correct, count);

    }
    catch (cl::Error err) {
        std::cout << "Exception\n";
        std::cerr
            << "ERROR: "
            << err.what()
            << "("
            << err_code(err.err())
           << ")"
           << std::endl;
    }
}
