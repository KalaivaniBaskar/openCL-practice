// summing integers with cahin addition

__kernel void vadd(                             
   __global int* a,                      
   __global int* b,                      
   __global int* c,                      
   const unsigned int count)               
{                                          
   int i = get_global_id(0);               
   if(i < count)  {
       c[i] = a[i] + b[i];                 
   }
}                                          