#define __CL_ENABLE_EXCEPTIONS

#include "cl.hpp"

#include "util.hpp" // utility library

#include "err_code.h"

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

//------------------------------------------------------------------------------

#define TOL    (0.001)   // tolerance used in floating point comparisons
#define LENGTH (9)    // length of vectors a, b, and c

void display(std::vector<float> &ip, int N){
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << ip[i * N + j] << " ";
        }
        std::cout << std::endl;
        }
}
int main(void)
{   
    int N = 3;
    std::vector<float> h_a(LENGTH);                // a vector 
    std::vector<float> h_b(LENGTH);                // b vector 	
    std::vector<float> h_c (LENGTH, 0xdeadbeef);    //
    std::vector<float> h_d (LENGTH, 0xdeadbeef);    // 
    std::vector<float> h_e (LENGTH, 0xdeadbeef);    // 

    cl::Buffer d_a;                        // device memory used for the input  a vector
    cl::Buffer d_b;                        // device memory used for the input  b vector
    cl::Buffer d_c;                       // device memory used for the # of work item oper
    cl::Buffer d_d;                       // device memory used for see the sum of global id
    cl::Buffer d_e;                       // device memory used for op of c = a * b mmul

    // Fill vectors a and b with random float values
    int count = N * N;
    for(int i = 0; i < count; i++)
    {
        h_a[i]  = 3.0f;
        h_b[i]  = 5.0f;
        
    } 
      std::cout << "Display matrix A :"<< std::endl;
      display(h_a, N);
      std::cout << "Display matrix B :"<< std::endl;
      display(h_b, N);
      std::cout << "Order of matrix : " << N << std::endl;  
     
    try 
    {
    	// Create a context
        cl::Context context(DEVICE);
        std::cout<< "Context for Device : " << DEVICE << std::endl;
        // Load in kernel source, creating a program object for the context

       // naive mmul
    //    cl::Program program(context, util::loadProgram("workid.cl"), true);
       
       // c row mmul 
        cl::Program program(context, util::loadProgram("C_row.cl"), true);

        // Get the command queue
        cl::CommandQueue queue(context);

        // Create the kernel functor
 
    //    auto workid = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int>(program, "workid");
   
        // ^ whenever this functor 'workid' is called, the kernel executes with args passed with it.

        d_a   = cl::Buffer(context, begin(h_a), end(h_a), true);
        d_b   = cl::Buffer(context, begin(h_b), end(h_b), true);

        d_c  = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * LENGTH);
        
        // lets store the sum of dim global id in this vector 
        d_d  = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * LENGTH); 

        // lets store the mmul op
        d_e  = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * LENGTH);


        util::Timer timer;

          // Do the multiplication COUNT times
        // for (int i = 0; i < 1; i++)
        // {
        

        // for naive mmul
        // cl::NDRange global(N, N);
        // workid(
        //     cl::EnqueueArgs(
        //         queue,
        //         global), 
        //     d_a,
        //     d_b,
        //     d_c,
        //     d_d,
        //     d_e,
        //     N); 

        // for mmul per row : 

           // Create the compute kernel from the program 
           auto crow_mmul = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int>(program, "mmul");

          cl::NDRange global(N);
          // 1D ND Range set to number of rows in the C matrix 
          // no of work items = no of rows in op C matrix
          
          crow_mmul(cl::EnqueueArgs(queue, global),
                    d_a, d_b, d_c, d_d, d_e, N);

        queue.finish();

        double rtime = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
        printf("\nThe kernels ran in %lf seconds\n", rtime);

        cl::copy(queue, d_c, begin(h_c), end(h_c));
        cl::copy(queue, d_d, begin(h_d), end(h_d));
        cl::copy(queue, d_e, begin(h_e), end(h_e));
        // }
        // Test the results 
         std::cout << "o/p check E =  " << h_e[0] << " from: " << h_a[0] << " and " << h_b[0] << " for matrix order " << N << std::endl;

        // Display result:
        std::cout << "\nResultant Matrix D(sum of workid i+j):" << std::endl;
        for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << h_d[i * N + j] << " ";
        }
        std::cout << std::endl;
        }

        // Display result:
        std::cout << "\nResultant Matrix C(no of times each work item ran):" << std::endl;
        for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << h_c[i * N + j] << " ";
        }
        std::cout << std::endl;
        }
        // Display result:
        std::cout << "\nResultant Matrix E(e = a * b):" << std::endl;
        for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << h_e[i * N + j] << " ";
        }
        std::cout << std::endl;
        }
       
       // ignore below code for mmul. it is for vadd
        int correct = 0;
        float tmp;
        for(int i = 0; i < count; i++) {
            tmp = h_a[i] + h_b[i]; // expected value for d_c[i]
            tmp -= h_c[i];                      // compute errors
            if(tmp*tmp < TOL*TOL) {      // correct if square deviation is less 
                correct++;                         //  than tolerance squared
            }
            else {

                printf(
                    " tmp %f h_a %f h_b %f  h_c %f \n",
                    tmp, 
                    h_a[i], 
                    h_b[i], 
                    h_c[i]);
            }
        }

        // summarize results
        printf(
            "vector add to find C = A+B:  %d out of %d results were correct.\n", 
            correct, 
            count);
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
