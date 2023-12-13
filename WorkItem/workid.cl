// Purpose: see how many times each wirk item runs

__kernel void workid(                             
   __global float* a,   
   __global float* b,                                    
   __global float* c,     
   __global float* d, 
   __global float* e,                      
   const int N)               
{                                          
   int i = get_global_id(0);      
   int j = get_global_id(1);
   float tmp;    
   float tmpp;    
   int k;
   if ((i < N) && (j < N)) {
      tmp = 0.0f;
      tmpp = 0.0f;
      for(k=0;k<N;k++) {
         tmp = tmp + 1.0f;
         tmpp += a[i*N+k] * b[k*N+j];
      }  
       c[i*N+j] = tmp;     
       d[i*N+j] = i+j;      
       e[i*N+j] = tmpp;           
   }
}                                          