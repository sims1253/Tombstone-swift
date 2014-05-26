/**
 * @brief A global, B column private
 * @details [long description]
 * 
 * @param rowDim [description]
 * @param colDim [description]
 * @param common [description]
 * @param T [description]
 * @param T [description]
 * @param T [description]
 * @param T [description]
 */
 #define BLOCK_SIZE 16

 template<class T>
__kernel void multiLocal(const int ni, const int nj, const int nk, 
	__global T *A, __global T *B, __global T *C)
{
//     
int gj = get_global_id(0);    int gi = get_global_id(1); 
int bj = get_group_id(0);     int bi = get_group_id(1);  // Block index
int tj = get_local_id(0);     int ti = get_local_id(1);  // Thread index
int oj = bi*16;       int oi = bj*16; 
T Csub =0; 
__local T As   [16][16];
__local T Bs   [16][16];
	for (int ok = 0; ok < nk; ok += 16 ){
	    As[ti][tj] = A[ nk*(gi   ) + tj + ok ];   // A[i][k]
	    Bs[ti][tj] = B[ nj*(ti+ok) + gj ];        // B[k][j]
	    barrier(CLK_LOCAL_MEM_FENCE);
	    for (int k = 0; k < 16; ++k){
	    	 Csub += As[ti][k] * Bs[k][tj];
	    }
	    barrier(CLK_LOCAL_MEM_FENCE);
	}
	C[ nj * ( gi ) + gj ] = Csub;
}