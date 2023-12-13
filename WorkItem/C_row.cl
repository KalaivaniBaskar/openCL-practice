// 1D ND Range set to number of rows in the C matrix 
// no of work items = no of rows in op C matrix

__kernel void mmul(
    __global float* A,
    __global float* B,
    __global float* C,
    __global float* D,
    __global float* E,   
      const int N)

{
    int k, j;
    int i = get_global_id(0);
    float tmp = 0.0f;
    float tmpp = 0.0f;
    if (i < N) {
          for (j = 0; j < N; j++) {
            tmp = 0.0f;
            tmpp = 0.0f;
            for (k = 0; k < N; k++){
                tmp += A[i*N+k] * B[k*N+j];
                tmpp = tmpp + 1.0f;
            }
            E[i*N+j] = tmp;
            D[i*N+j] = i+j;    
            C[i*N+j] = tmpp;   
        }
       
    }
}
