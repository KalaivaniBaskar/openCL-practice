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
    std::vector<int> h_a(LENGTH);                // a vector 
    std::vector<int> h_b(LENGTH);                // b vector 	
    std::vector<int> h_c (LENGTH, 0xdeadbeef);   // c vector (result)
    std::vector<int> h_d (LENGTH, 0xdeadbeef);   // d vector (result)
    std::vector<int> h_e (LENGTH);               // e vector
    std::vector<int> h_f (LENGTH, 0xdeadbeef);   // f vector (result)
    std::vector<int> h_g (LENGTH);               // g vector

    cl::Buffer d_a;                       // device memory used for the input  a vector
    cl::Buffer d_b;                       // device memory used for the input  b vector
    cl::Buffer d_c;                       // device memory used for the output c vector
    cl::Buffer d_d;                       // device memory used for the output d vector
    cl::Buffer d_e;                       // device memory used for the input e vector
    cl::Buffer d_f;                       // device memory used for the output f vector
    cl::Buffer d_g;                       // device memory used for the input g vector

    // Fill vectors a and b with random INT values
    int count = LENGTH;
    for(int i = 0; i < count; i++)
    {
        h_a[i]  = rand() % 100;
        h_b[i]  = rand() % 100;
        h_e[i]  = rand() % 100;
        h_g[i]  = rand() % 100;
    }

    try 
    {
    	// Create a context
        cl::Context context(DEVICE);

        // Load in kernel source, creating a program object for the context

        cl::Program program(context, util::loadProgram("vadd_chain.cl"), true);

        // Get the command queue
        cl::CommandQueue queue(context);

        // Create the kernel functor
 
        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int> vadd(program, "vadd");

        d_a   = cl::Buffer(context, h_a.begin(), h_a.end(), true);
        d_b   = cl::Buffer(context, h_b.begin(), h_b.end(), true);
        d_e   = cl::Buffer(context, h_e.begin(), h_e.end(), true);
        d_g   = cl::Buffer(context, h_g.begin(), h_g.end(), true);

        // c stores a + b
        d_c  = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(int) * LENGTH); 
        // d stores e + c
        d_d  = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(int) * LENGTH);
        // f stores g + d 
        // so f = a + b + e + g
        d_f  = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * LENGTH);

        util::Timer timer;

        vadd(
            cl::EnqueueArgs(
                queue,
                cl::NDRange(count)), 
            d_a,
            d_b,
            d_c,
            count);

        vadd(
            cl::EnqueueArgs(
                queue,
                cl::NDRange(count)), 
            d_e,
            d_c,
            d_d,
            count);

        vadd(
            cl::EnqueueArgs(
                queue,
                cl::NDRange(count)), 
            d_g,
            d_d,
            d_f,
            count);

        double rtime = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
        printf("\nThe kernels ran in %lf seconds\n", rtime);
        
        // for checking purpose, copying intermediate sum to host, not necessary
        cl::copy(queue, d_c, h_c.begin(), h_c.end());
        cl::copy(queue, d_d, h_d.begin(), h_d.end());

        // ccopying final result to host
        cl::copy(queue, d_f, h_f.begin(), h_f.end());

      
        // Test the results 
        std::cout << "o/p check F = A+B+E+G : " << std::endl;
        std::cout<< "C=A+B is " << h_c[0] << " from: " << h_a[0] << " and " << h_b[0] << std::endl;
        std::cout<< "D=C+E is " << h_d[0] << " from: " << h_c[0] << " and " << h_e[0] << std::endl;
        std::cout<< "F=D+G is " << h_f[0] << " from: " << h_d[0] << " and " << h_g[0] << std::endl;
        std::cout << "output f : " << h_f[0] << std::endl;
        std::cout << "from A,B,E,G: " << h_a[0] << " and " << h_b[0] << " and "  << h_e[0] << " and "  << h_g[0] << std::endl;
        // Test the results
        int correct = 0;
        int tmp;
        for(int i = 0; i < count; i++)
        {
            tmp = h_a[i] + h_b[i] + h_e[i] + h_g[i];     // assign element i of a+b+e+g to tmp
            tmp -= h_f[i];                               // compute deviation of expected and output result
            if(tmp*tmp < TOL*TOL)                        // correct if square deviation is less than tolerance squared
                correct++;
            else {
                printf(" tmp %d h_a %d h_b %d h_e %d h_g %d h_f %d\n",tmp, h_a[i], h_b[i], h_e[i], h_g[i], h_f[i]);
            }
        }
        
        // summarize results
        printf("C = A+B+E+G:  %d out of %d results were correct.\n", correct, count);
        
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
