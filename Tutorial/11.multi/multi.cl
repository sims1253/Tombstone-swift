template<class T>
__kernel void multi(const int rowDim, const int colDim, const int productDim, __global T *A, __global T *B, __global T *C)
{
	int i,j,k;

	i=get_global_id(0);
	j=get_global_id(1);

	for(k=0; k<productDim;k++){
		C[i*colDim+j]+=A[i*colDim+k]*B[k*productDim+j];
	}
}

