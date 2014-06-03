/*
 * Matrix multiplication: C = A * B.
 * Device code.
 */

// Thread block size
//#define BLOCK_SIZE 16

//////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! wA is A's width and wB is B's width
//////////////////////////////////////////////////////
template<class T>
__kernel void GpuMatrixMul( __global T* A, __global T* B,__global T* C, const int wA, const int wB)
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

     // T tmp[4];
     // tmp[0]=0.0f;
     // tmp[1]=0.0f;
     // tmp[2]=0.0f;
     // tmp[3]=0.0f;
     // T sumPatch[2];

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

            // tmp[0] += As[ty][k]    * Bs[k][tx];
            // tmp[1] += As[ty][k+4]  * Bs[k+4][tx];
            // tmp[2] += As[ty][k+8]  * Bs[k+8][tx];
            // tmp[3] += As[ty][k+12] * Bs[k+12][tx];

           Csub +=As[ty][k]    * Bs[k][tx];

        }

        // read_mem_fence(CLK_LOCAL_MEM_FENCE);
        // sumPatch[0] = tmp[0]+tmp[1];
        // sumPatch[1] = tmp[2]+tmp[3];
        // read_mem_fence(CLK_LOCAL_MEM_FENCE);
        // Csub = Csub + sumPatch[0];
        // Csub = Csub + sumPatch[1];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        barrier(CLK_LOCAL_MEM_FENCE);

    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * 16 * by + 16 * bx;
    C[c + wB * ty + tx] = Csub;


}
