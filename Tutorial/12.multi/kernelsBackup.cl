// Matrix multiplication, column major format
template<class T>
__kernel void CMrolled( __global T* C, __global T* A,__global T* B, const int wA, const int wB, const int hA)
{
    const int hB = wA;
    // Block index
    int bx = get_group_id(0);
    int by = get_group_id(1);
 
    // Thread index
    int tx = get_local_id(0);
    int ty = get_local_id(1);
 
    // Index of the first sub-matrix of A processed 
    // by the block
    /* nur noch BLOCK_SIZE * by nötig da, matrix in column major  */
    int aBegin = 16 * by;
 
 
    // Step size used to iterate through the 
    // sub-matrices of A
    /* Es wird spaltenweise traversiert, daher muss aStep nun größer sein  */
    int aStep  = 16 * hA;
 
    // Index of the first sub-matrix of B processed 
    // by the block
    /*Bleibt das gleich oder nicht?!  */
    int bBegin = 16 * bx;
 
    // Step size used to iterate through the 
    // sub-matrices of B
    /* Spaltenweises Traversieren benötigt für B einen kleineren Step */
    int bStep  = 16;

    int bEnd = bBegin + hB -1;

    /* Lokale Matrizen werden wie bei row-major aufbereitet!! */


    // Declaration of the local memory array As 
    // used to store the sub-matrix of A
    __local T As[16][16];
 
    // Declaration of the local memory array Bs 
    // used to store the sub-matrix of B
    __local T Bs[16][16];

    T Csub = 0.0f;

 
    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin; b <= bEnd; a += aStep, b += bStep) 
    {

        // Load the matrices from global memory
        // to local memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + ty + hA * tx];
        Bs[ty][tx] = B[b + ty + hA * tx];

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
    /* Ergebnismatrix muss nun auch in column-major Format abgespeichert werden */
    int c = 16 * ( by + hA* bx);
    C[c + hA * tx + ty] = Csub;
   

}


// Matrix multiplication, column major format
template<class T>
__kernel void CMunrolled( __global T* C, __global T* A,__global T* B, const int wA, const int wB, const int hA)
{
    const int hB = wA;
    // Block index
    int bx = get_group_id(0);
    int by = get_group_id(1);
 
    // Thread index
    int tx = get_local_id(0);
    int ty = get_local_id(1);
 
    // Index of the first sub-matrix of A processed 
    // by the block
    /* nur noch BLOCK_SIZE * by nötig da, matrix in column major  */
    int aBegin = 32 * by;
 
 
    // Step size used to iterate through the 
    // sub-matrices of A
    /* Es wird spaltenweise traversiert, daher muss aStep nun größer sein  */
    int aStep  = 32 * hA;

    int aEnd = aBegin + (wA * hA) -1;
 
    // Index of the first sub-matrix of B processed 
    // by the block
    /*Bleibt das gleich oder nicht?!  */
    int bBegin = 32 * hA * bx;
 
    // Step size used to iterate through the 
    // sub-matrices of B
    /* Spaltenweises Traversieren benötigt für B einen kleineren Step */
    int bStep  = 32;

    //int bEnd = bBegin + hB -1 ;

    /* Lokale Matrizen werden wie bei row-major aufbereitet!! */


    // Declaration of the local memory array As 
    // used to store the sub-matrix of A
    __local T As[32][32];
 
    // Declaration of the local memory array Bs 
    // used to store the sub-matrix of B
    __local T Bs[32][32];

    T Csub [16]= {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

 
    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) 
    {

        // Load the matrices from global memory
        // to local memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + ty + (tx * hA)];
        Bs[ty][tx] = B[b + ty + (tx * hB)];

        As[ty+2][tx] = A[a + ty+2 + (tx * hA)];
        Bs[ty+2][tx] = B[b + ty+2 + (tx * hB)];

        As[ty+4][tx] = A[a + ty+4 + (tx * hA)];
        Bs[ty+4][tx] = B[b + ty+4 + (tx * hB)];

        As[ty+6][tx] = A[a + ty+6 + (tx * hA)];
        Bs[ty+6][tx] = B[b + ty+6 + (tx * hB)];

        As[ty+8][tx] = A[a + ty+8 + (tx * hA)];
        Bs[ty+8][tx] = B[b + ty+8 + (tx * hB)];

        As[ty+10][tx] = A[a + ty+10 + (tx * hA)];
        Bs[ty+10][tx] = B[b + ty+10 + (tx * hB)];

        As[ty+12][tx] = A[a + ty+12 + (tx * hA)];
        Bs[ty+12][tx] = B[b + ty+12 + (tx * hB)];

        As[ty+14][tx] = A[a + ty+14 + (tx * hA)];
        Bs[ty+14][tx] = B[b + ty+14 + (tx * hB)];

        As[ty+16][tx] = A[a + ty+16 + (tx * hA)];
        Bs[ty+16][tx] = B[b + ty+16 + (tx * hB)];

        As[ty+18][tx] = A[a + ty+18 + (tx * hA)];
        Bs[ty+18][tx] = B[b + ty+18 + (tx * hB)];

        As[ty+20][tx] = A[a + ty+20 + (tx * hA)];
        Bs[ty+20][tx] = B[b + ty+20 + (tx * hB)];

        As[ty+22][tx] = A[a + ty+22 + (tx * hA)];
        Bs[ty+22][tx] = B[b + ty+22 + (tx * hB)];

        As[ty+24][tx] = A[a + ty+24 + (tx * hA)];
        Bs[ty+24][tx] = B[b + ty+24 + (tx * hB)];

        As[ty+26][tx] = A[a + ty+26 + (tx * hA)];
        Bs[ty+26][tx] = B[b + ty+26 + (tx * hB)];

        As[ty+28][tx] = A[a + ty+28 + (tx * hA)];
        Bs[ty+28][tx] = B[b + ty+28 + (tx * hB)];

        As[ty+30][tx] = A[a + ty+30 + (tx * hA)];
        Bs[ty+30][tx] = B[b + ty+30 + (tx * hB)];

        // Synchronize to make sure the matrices 
        // are loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < 32; ++k){
            T Btemp = Bs[k][tx];
            Csub[0] +=As[ty][k]    * Btemp;
            Csub[1] +=As[ty+2][k]    * Btemp;
            Csub[2] +=As[ty+4][k]    * Btemp;
            Csub[3] +=As[ty+6][k]    * Btemp;
            Csub[4] +=As[ty+8][k]    * Btemp;
            Csub[5] +=As[ty+10][k]    * Btemp;
            Csub[6] +=As[ty+12][k]    * Btemp;
            Csub[7] +=As[ty+14][k]    * Btemp;
            Csub[8] +=As[ty+16][k]    * Btemp;
            Csub[9] +=As[ty+18][k]    * Btemp;
            Csub[10] +=As[ty+20][k]    * Btemp;
            Csub[11] +=As[ty+22][k]    * Btemp;
            Csub[12] +=As[ty+24][k]    * Btemp;
            Csub[13] +=As[ty+26][k]    * Btemp;
            Csub[14] +=As[ty+28][k]    * Btemp;
            Csub[15] +=As[ty+30][k]    * Btemp;
       
        }

 
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        barrier(CLK_LOCAL_MEM_FENCE);
 
    }
 
    // Write the block sub-matrix to device memory;
    // each thread writes one element
    /* Ergebnismatrix muss nun auch in column-major Format abgespeichert werden */
   int c = 32 * by + 32 * hA * bx;
    C[c + ty + tx * hA] = Csub[0];
    C[c + ty + 2 + (tx * hA)] = Csub[1];
    C[c + ty + 4 + (tx * hA)] = Csub[2];
    C[c + ty + 6 + (tx * hA)] = Csub[3];
    C[c + ty + 8 + (tx * hA)] = Csub[4];
    C[c + ty + 10 + (tx * hA)] = Csub[5];
    C[c + ty + 12 + (tx * hA)] = Csub[6];
    C[c + ty + 14 + (tx * hA)] = Csub[7];
    C[c + ty + 16 + (tx * hA)] = Csub[8];
    C[c + ty + 18 + (tx * hA)] = Csub[9];
    C[c + ty + 20 + (tx * hA)] = Csub[10];
    C[c + ty + 22 + (tx * hA)] = Csub[11];
    C[c + ty + 24 + (tx * hA)] = Csub[12];
    C[c + ty + 26 + (tx * hA)] = Csub[13];
    C[c + ty + 28 + (tx * hA)] = Csub[14];
    C[c + ty + 30 + (tx * hA)] = Csub[15];
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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
__kernel void RMunrolled( __global T* C, __global T* A,__global T* B, const int wA, const int wB, const int hA)
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

    T Csub [16]= {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

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
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        As[ty+2][tx] = A[a + wA * (ty+2) +tx];
        Bs[ty+2][tx] = B[b + wB * (ty+2) +tx];

        As[ty+4][tx] = A[a + wA * (ty+4) +tx];
        Bs[ty+4][tx] = B[b + wB * (ty+4) +tx];

        As[ty+6][tx] = A[a + wA * (ty+6) +tx];
        Bs[ty+6][tx] = B[b + wB * (ty+6) +tx];

        As[ty+8][tx] = A[a + wA * (ty+8) +tx];
        Bs[ty+8][tx] = B[b + wB * (ty+8) +tx];

        As[ty+10][tx] = A[a + wA * (ty+10) +tx];
        Bs[ty+10][tx] = B[b + wB * (ty+10) +tx];        

        As[ty+12][tx] = A[a + wA * (ty+12) +tx];
        Bs[ty+12][tx] = B[b + wB * (ty+12) +tx];

        As[ty+14][tx] = A[a + wA * (ty+14) +tx];
        Bs[ty+14][tx] = B[b + wB * (ty+14) +tx];

        As[ty+16][tx] = A[a + wA * (ty+16) +tx];
        Bs[ty+16][tx] = B[b + wB * (ty+16) +tx];

        As[ty+18][tx] = A[a + wA * (ty+18) +tx];
        Bs[ty+18][tx] = B[b + wB * (ty+18) +tx];

        As[ty+20][tx] = A[a + wA * (ty+20) +tx];
        Bs[ty+20][tx] = B[b + wB * (ty+20) +tx];

        As[ty+22][tx] = A[a + wA * (ty+22) +tx];
        Bs[ty+22][tx] = B[b + wB * (ty+22) +tx];

        As[ty+24][tx] = A[a + wA * (ty+24) +tx];
        Bs[ty+24][tx] = B[b + wB * (ty+24) +tx];

        As[ty+26][tx] = A[a + wA * (ty+26) +tx];
        Bs[ty+26][tx] = B[b + wB * (ty+26) +tx];

        As[ty+28][tx] = A[a + wA * (ty+28) +tx];
        Bs[ty+28][tx] = B[b + wB * (ty+28) +tx];

        As[ty+30][tx] = A[a + wA * (ty+30) +tx];
        Bs[ty+30][tx] = B[b + wB * (ty+30) +tx];

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


            Csub[0] +=As[ty][k]    * Bs[k][tx];
            Csub[1] +=As[ty+2][k]    * Bs[k][tx];
            Csub[2] +=As[ty+4][k]    * Bs[k][tx];
            Csub[3] +=As[ty+6][k]    * Bs[k][tx];
            Csub[4] +=As[ty+8][k]    * Bs[k][tx];
            Csub[5] +=As[ty+10][k]    * Bs[k][tx];
            Csub[6] +=As[ty+12][k]    * Bs[k][tx];
            Csub[7] +=As[ty+14][k]    * Bs[k][tx];
            Csub[8] +=As[ty+16][k]    * Bs[k][tx];
            Csub[9] +=As[ty+18][k]    * Bs[k][tx];
            Csub[10] +=As[ty+20][k]    * Bs[k][tx];
            Csub[11] +=As[ty+22][k]    * Bs[k][tx];
            Csub[12] +=As[ty+24][k]    * Bs[k][tx];
            Csub[13] +=As[ty+26][k]    * Bs[k][tx];
            Csub[14] +=As[ty+28][k]    * Bs[k][tx];
            Csub[15] +=As[ty+30][k]    * Bs[k][tx];
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
__kernel void RMunrolledBsregister( __global T* C, __global T* A,__global T* B, const int wA, const int wB, const int hA)
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
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        As[ty+2][tx] = A[a + wA * (ty+2) +tx];
        Bs[ty+2][tx] = B[b + wB * (ty+2) +tx];

        As[ty+4][tx] = A[a + wA * (ty+4) +tx];
        Bs[ty+4][tx] = B[b + wB * (ty+4) +tx];

        As[ty+6][tx] = A[a + wA * (ty+6) +tx];
        Bs[ty+6][tx] = B[b + wB * (ty+6) +tx];

        As[ty+8][tx] = A[a + wA * (ty+8) +tx];
        Bs[ty+8][tx] = B[b + wB * (ty+8) +tx];

        As[ty+10][tx] = A[a + wA * (ty+10) +tx];
        Bs[ty+10][tx] = B[b + wB * (ty+10) +tx];        

        As[ty+12][tx] = A[a + wA * (ty+12) +tx];
        Bs[ty+12][tx] = B[b + wB * (ty+12) +tx];

        As[ty+14][tx] = A[a + wA * (ty+14) +tx];
        Bs[ty+14][tx] = B[b + wB * (ty+14) +tx];

        As[ty+16][tx] = A[a + wA * (ty+16) +tx];
        Bs[ty+16][tx] = B[b + wB * (ty+16) +tx];

        As[ty+18][tx] = A[a + wA * (ty+18) +tx];
        Bs[ty+18][tx] = B[b + wB * (ty+18) +tx];

        As[ty+20][tx] = A[a + wA * (ty+20) +tx];
        Bs[ty+20][tx] = B[b + wB * (ty+20) +tx];

        As[ty+22][tx] = A[a + wA * (ty+22) +tx];
        Bs[ty+22][tx] = B[b + wB * (ty+22) +tx];

        As[ty+24][tx] = A[a + wA * (ty+24) +tx];
        Bs[ty+24][tx] = B[b + wB * (ty+24) +tx];

        As[ty+26][tx] = A[a + wA * (ty+26) +tx];
        Bs[ty+26][tx] = B[b + wB * (ty+26) +tx];

        As[ty+28][tx] = A[a + wA * (ty+28) +tx];
        Bs[ty+28][tx] = B[b + wB * (ty+28) +tx];

        As[ty+30][tx] = A[a + wA * (ty+30) +tx];
        Bs[ty+30][tx] = B[b + wB * (ty+30) +tx];

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
