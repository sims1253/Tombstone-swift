// Matrix multiplication, column major format
template<class T>
__kernel void GpuMatrixMulCM( __global T* C, __global T* A,__global T* B, const int wA, const int wB, const int hA)
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
    Csub[0] = 0.0f;
    Csub[1] = 0.0f;
    Csub[2] = 0.0f;
    Csub[3] = 0.0f;
    Csub[4] = 0.0f;
    Csub[5] = 0.0f;
    Csub[6] = 0.0f;
    Csub[7] = 0.0f;
    Csub[8] = 0.0f;
    Csub[9] = 0.0f;
    Csub[10] = 0.0f;
    Csub[11] = 0.0f;
    Csub[12] = 0.0f;
    Csub[13] = 0.0f;
    Csub[14] = 0.0f;
    Csub[15] = 0.0f;
}
