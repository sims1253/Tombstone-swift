/**
 * @brief A row in private memory, B column in local memory
 * @details OpenCL-Kernel for ffast matrix multiplication
 * 
 * @param rowDim 
 * @param colDim [description]
 * @param common [description]
 * @param T Matrix A
 * @param T Matrix B
 * @param T Matrix C
 * @param T Pointer for local memory
 * 
 */

template<class T>
__kernel void multiLocal(const int rowDim, const int colDim, const int common, 
	__global T *A, __global T *B, __global T *C)
{
int j,k;

int i=get_global_id(0);

//int iloc = get_local_id(0);
//int nloc = get_local_size(0);

T Aprivate[1024];
T temp;

//Row major Format
for(k=0;k<common;k++){
Aprivate[k] = A[i*colDim+k];
}

// copy from global memory to local memory
for(j=0; j<colDim;j++){
	temp=0.0f;
//for(k=iloc;k<common;k+=nloc){
//BlocalMem[k]=B[k*common+j];
//}


// wait until copying finished
//barrier(CLK_LOCAL_MEM_FENCE);



// calculates
for(k=0; k<common;k++){
temp+=Aprivate[k]*B[k*common+j];
}
C[i*colDim+j] += temp;

}
}
