//------------------------------------------------------------------------------
//
//  PROGRAM: Matrix Multiplication driver
//
//------------------------------------------------------------------------------

#include "matmul.hpp"
#include "matrix_lib.hpp"
#include "util.hpp"
#include "err_code.h"
#include "device_picker.hpp"


std::string kernelsource = "__kernel void mmul(                                                    \n" \
"   const int N,                                                        \n" \
"   __global float* A,                                                  \n" \
"   __global float* B,                                                  \n" \
"   __global float* C)                                                  \n" \
"{                                                                      \n" \
"   int k;                                                              \n" \
"   int i = get_global_id(0);                                           \n" \
"   int j = get_global_id(1);                                           \n" \
"   float tmp;                                                          \n" \
"   if ( (i < N) && (j <N))                                             \n" \
"   {                                                                   \n" \
"       tmp = 0.0;                                                      \n" \
"       for(k=0;k<N;k++)                                                \n" \
"           tmp += A[i*N+k] * B[k*N+j];                                 \n" \
"       C[i*N+j] = tmp;                                                 \n" \
"   }                                                                   \n" \
"}                                                                      \n" \
"\n";

int main(int argc, char *argv[])
{

    int N;                  // A[N][N], B[N][N], C[N][N]
    int size;               // Number of elements in each matrix


    double start_time;      // Starting time
    double run_time;        // Timing
    util::Timer timer;      // Timing

    N    = ORDER;   // no of rows/cols of matrix
    size = N * N; // a 3 order matrix will have 3*3 9 elements in a flat 1d array

    std::vector<float> h_A(size); // Host memory for Matrix A 
    std::vector<float> h_B(size); // Host memory for Matrix B
    std::vector<float> h_C(size); // Host memory for Matrix C

    cl::Buffer d_a, d_b, d_c;   // Matrices in device memory

//--------------------------------------------------------------------------------
// Create a context and queue
//--------------------------------------------------------------------------------

    try
    {

        cl_uint deviceIndex = 0;
        parseArguments(argc, argv, &deviceIndex);

        // Get list of devices
        std::vector<cl::Device> devices;
        unsigned numDevices = getDeviceList(devices);

        // Check device index in range
        if (deviceIndex >= numDevices)
        {
          std::cout << "Invalid device index (try '--list')\n";
          return EXIT_FAILURE;
        }

        cl::Device device = devices[deviceIndex];

        std::string name;
        getDeviceName(device, name);
        std::cout << "\nUsing OpenCL device: " << name << "\n";

        std::vector<cl::Device> chosen_device;
        chosen_device.push_back(device);
        cl::Context context(chosen_device);
        cl::CommandQueue queue(context, device);

//--------------------------------------------------------------------------------
// Run sequential matmul
//--------------------------------------------------------------------------------


        initmat(N, h_A, h_B, h_C);

        timer.reset();

        printf("\n===== Sequential, matrix mult (dot prod), order %d on host CPU ======\n",N);
        for(int i = 0; i < COUNT; i++)
        {
            zero_mat(N, h_C);

            start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;

            seq_mat_mul_sdot(N, h_A, h_B, h_C);

            run_time  = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0 - start_time;
            results(N, h_C, run_time);
        }
        // Display result:
        std::cout << "\nResultant Matrix C(seq mmul):" << std::endl;
        for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
//--------------------------------------------------------------------------------
// Setup the buffers, initialize matrices, and write them into global memory
//--------------------------------------------------------------------------------

        //  Reset A, B and C matrices (just to play it safe)
        initmat(N, h_A, h_B, h_C);

        d_a = cl::Buffer(context, h_A.begin(), h_A.end(), true);

        d_b = cl::Buffer(context, h_B.begin(), h_B.end(), true);

        d_c = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * size);

//--------------------------------------------------------------------------------
// OpenCL matrix multiplication ... Naive
//--------------------------------------------------------------------------------

        // Create the compute program from the source buffer
        cl::Program program(context, kernelsource, true);

        // Create the compute kernel from the program
        cl::make_kernel<int, cl::Buffer, cl::Buffer, cl::Buffer> naive_mmul(program, "mmul");

        printf("\n===== OpenCL, matrix mult, C(i,j) per work item, order %d ======\n",N);

         // Display the matrices
        std::cout << "Matrix A:" << std::endl;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                std::cout << h_A[i * N + j] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "\nMatrix B:" << std::endl;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                std::cout << h_B[i * N + j] << " ";
            }
            std::cout << std::endl;
        }
  
        // Do the multiplication COUNT times
        for (int i = 0; i < COUNT; i++)
        {   
            std::cout<<"kernel called "<< i << "\n";
            zero_mat(N, h_C);

            start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;

            // Execute the kernel over the entire range of C matrix elements ... computing
            // a dot product for each element of the product matrix.  The local work
            // group size is set to NULL ... so I'm telling the OpenCL runtime to
            // figure out a local work group size for me.
            cl::NDRange global(N, N);
            naive_mmul(cl::EnqueueArgs(queue, global),
                    N, d_a, d_b, d_c);

            queue.finish();

            run_time  = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0 - start_time;

            cl::copy(queue, d_c, h_C.begin(), h_C.end());

            results(N, h_C, run_time);

        } // end for loop

        // Display result:
        std::cout << "\nResultant Matrix C(openCL):" << std::endl;
        for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }


    } catch (cl::Error err)
    {
        std::cout << "Exception\n";
        std::cerr << "ERROR: "
                  << err.what()
                  << "("
                  << err_code(err.err())
                  << ")"
                  << std::endl;
    }

    return EXIT_SUCCESS;
}
