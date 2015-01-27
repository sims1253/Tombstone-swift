/*
 * Matrix multiplication: C = A * B.
 * Device code.
 */

// Thread block size
//#define BLOCK_SIZE 32

//////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! wA is A's width and wB is B's width
//////////////////////////////////////////////////////
template<class T>
__kernel void RMBetterMem( __global T* C, __global T* A,__global T* B, const int wA, const int wB, const int hA)
{

    // Block index
    int bx = get_group_id(0);
    int by = get_group_id(1);

    // Thread index
    int tx = get_local_id(0);
    int ty = get_local_id(1);

    // Index of the first sub-matrix of A processed
    // by the block
    int aBegin = wA * 32 * by;

    // Index of the last sub-matrix of A processed
    // by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the
    // sub-matrices of A
    int aStep  = 32;

    // Index of the first sub-matrix of B processed
    // by the block
    int bBegin = 32 * bx;

    // Step size used to iterate through the
    // sub-matrices of B
    int bStep  = 32 * wB;

    // Declaration of the local memory array As
    // used to store the sub-matrix of A
   // __local T As[32][32];

    // Declaration of the local memory array Bs
    // used to store the sub-matrix of B
    //__local T Bs[32][32];

    T Csub [16]= {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

     //T tmp[2];
     //T sumPatch[2];

     //tmp[0]=0.0f;
     //tmp[1]=0.0f;
     /*
     tmp[2]=0.0f;
     tmp[3]=0.0f;
     sumPatch[0]=0.0f;
     sumPatch[1]=0.0f;
     */
    __local T As[32][32];
    __local T Bs[32][32];
    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep)
    {

        // Load the matrices from global memory
        // to local memory; each thread loads
        // one element of each matrix
        As[ty*16][tx] = A[a + wA * ty + tx];
        Bs[ty*16][tx] = B[b + wB * ty + tx];

        As[1+(ty*16)][tx] = A[a + wA * (1+(ty*16)) +tx];
        Bs[1+(ty*16)][tx] = B[b + wB * (1+(ty*16)) +tx];

        As[2+(ty*16)][tx] = A[a + wA * (2+(ty*16)) +tx];
        Bs[2+(ty*16)][tx] = B[b + wB * (2+(ty*16)) +tx];

        As[3+(ty*16)][tx] = A[a + wA * (3+(ty*16)) +tx];
        Bs[3+(ty*16)][tx] = B[b + wB * (3+(ty*16)) +tx];

        As[4+(ty*16)][tx] = A[a + wA * (4+(ty*16)) +tx];
        Bs[4+(ty*16)][tx] = B[b + wB * (4+(ty*16)) +tx];

        As[5+(ty*16)][tx] = A[a + wA * (5+(ty*16)) +tx];
        Bs[5+(ty*16)][tx] = B[b + wB * (5+(ty*16)) +tx];        

        As[6+(ty*16)][tx] = A[a + wA * (6+(ty*16)) +tx];
        Bs[6+(ty*16)][tx] = B[b + wB * (6+(ty*16)) +tx];

        As[7+(ty*16)][tx] = A[a + wA * (7+(ty*16)) +tx];
        Bs[7+(ty*16)][tx] = B[b + wB * (7+(ty*16)) +tx];

        As[8+(ty*16)][tx] = A[a + wA * (8+(ty*16)) +tx];
        Bs[8+(ty*16)][tx] = B[b + wB * (8+(ty*16)) +tx];

        As[9+(ty*16)][tx] = A[a + wA * (9+(ty*16)) +tx];
        Bs[9+(ty*16)][tx] = B[b + wB * (9+(ty*16)) +tx];

        As[10+(ty*16)][tx] = A[a + wA * (10+(ty*16)) +tx];
        Bs[10+(ty*16)][tx] = B[b + wB * (10+(ty*16)) +tx];

        As[11+(ty*16)][tx] = A[a + wA * (11+(ty*16)) +tx];
        Bs[11+(ty*16)][tx] = B[b + wB * (11+(ty*16)) +tx];

        As[12+(ty*16)][tx] = A[a + wA * (12+(ty*16)) +tx];
        Bs[12+(ty*16)][tx] = B[b + wB * (12+(ty*16)) +tx];

        As[13+(ty*16)][tx] = A[a + wA * (13+(ty*16)) +tx];
        Bs[13+(ty*16)][tx] = B[b + wB * (13+(ty*16)) +tx];

        As[14+(ty*16)][tx] = A[a + wA * (14+(ty*16)) +tx];
        Bs[14+(ty*16)][tx] = B[b + wB * (14+(ty*16)) +tx];

        As[15+(ty*16)][tx] = A[a + wA * (15+(ty*16)) +tx];
        Bs[15+(ty*16)][tx] = B[b + wB * (15+(ty*16)) +tx];

        //tmp[0]=0.0f;
        //tmp[1]=0.0f;
        /*
        tmp[2]=0.0f;
        tmp[3]=0.0f;
        sumPatch[0]=0.0f;
        sumPatch[1]=0.0f;
        */


        // Synchronize to make sure the matrices
        // are loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        #pragma unroll
        for (int k = 0; k < 32; ++k){

             //tmp[0] += As[ty][k]    * Bs[k][tx];

             //tmp[1] += As[ty][k+8]  * Bs[k+8][tx];
            T btemp = Bs[k][tx];

            Csub[0] +=As[ty][k]    * btemp;
            Csub[1] +=As[ty+2][k]    * btemp;
            Csub[2] +=As[ty+4][k]    * btemp;
            Csub[3] +=As[ty+6][k]    * btemp;
            Csub[4] +=As[ty+8][k]    * btemp;
            Csub[5] +=As[ty+10][k]    * btemp;
            Csub[6] +=As[ty+12][k]    * btemp;
            Csub[7] +=As[ty+14][k]    * btemp;
            Csub[8] +=As[ty+16][k]    * btemp;
            Csub[9] +=As[ty+18][k]    * btemp;
            Csub[10] +=As[ty+20][k]    * btemp;
            Csub[11] +=As[ty+22][k]    * btemp;
            Csub[12] +=As[ty+24][k]    * btemp;
            Csub[13] +=As[ty+26][k]    * btemp;
            Csub[14] +=As[ty+28][k]    * btemp;
            Csub[15] +=As[ty+30][k]    * btemp;
        }


         //Csub = tmp[1] + tmp[0];


        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        barrier(CLK_LOCAL_MEM_FENCE);

    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * 32 * by + 32 * bx;
    C[c + wB * ty + tx] = Csub[0];
    C[c + wB * (ty+2) + tx] = Csub[1];
    C[c + wB * (ty+4) + tx] = Csub[2];
    C[c + wB * (ty+6) + tx] = Csub[3];
    C[c + wB * (ty+8) + tx] = Csub[4];
    C[c + wB * (ty+10) + tx] = Csub[5];
    C[c + wB * (ty+12) + tx] = Csub[6];
    C[c + wB * (ty+14) + tx] = Csub[7];
    C[c + wB * (ty+16) + tx] = Csub[8];
    C[c + wB * (ty+18) + tx] = Csub[9];
    C[c + wB * (ty+20) + tx] = Csub[10];
    C[c + wB * (ty+22) + tx] = Csub[11];
    C[c + wB * (ty+24) + tx] = Csub[12];
    C[c + wB * (ty+26) + tx] = Csub[13];
    C[c + wB * (ty+28) + tx] = Csub[14];
    C[c + wB * (ty+30) + tx] = Csub[15];
}
