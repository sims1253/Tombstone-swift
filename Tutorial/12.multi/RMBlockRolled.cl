/*
 * Matrix multiplication: C = A * B.
 * Device code.
 */
 

// Matrices are stored in row-major format
// Thread block size
//#define BLOCK_SIZE 16
  
//////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! wA is A's width and wB is B's width
//////////////////////////////////////////////////////
template<class T>
__kernel void RMBlockRolled( __global T* A, __global T* B,__global T* C, const int wA, const int wB)
{
    // Block index
    int bx = get_group_id(0);
    int by = get_group_id(1);
 
    // Thread index
    int tx = get_local_id(0);
    int ty = get_local_id(1);
 
    // Index of the first sub-matrix of A processed 
    // by the block
    int aBegin = wA * 16 * by;
 
    // Index of the last sub-matrix of A processed 
    // by the block
    int aEnd   = aBegin + wA - 1;
 
    // Step size used to iterate through the 
    // sub-matrices of A
    int aStep  = 16;
 
    // Index of the first sub-matrix of B processed 
    // by the block
    int bBegin = 16 * bx;
 
    // Step size used to iterate through the 
    // sub-matrices of B
    int bStep  = 16 * wB;

    // Declaration of the local memory array As 
    // used to store the sub-matrix of A
    __local T As[16][16];
 
    // Declaration of the local memory array Bs 
    // used to store the sub-matrix of B
    __local T Bs[16][16];

    T Csub = 0.0f;

 
    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) 
    {

        // Load the matrices from global memory
        // to local memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices 
        // are loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < 16; ++k){
            Csub += As[ty][k] * Bs[k][tx];
       
        }

 
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        barrier(CLK_LOCAL_MEM_FENCE);
 
    }
 
    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = 16 * ( wB * by + bx);
    C[c + wB * ty + tx] = Csub;
   

}