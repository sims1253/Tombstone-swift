template<class T>
__kernel void multiLocal(const int rowDim, const int colDim, const int common, __global T *A, __global T *B, __global T *C, __local, *T BlocalMem)
{
	int j,k;

	int i=get_global_id(0);
	int iloc = get_local_id(0);
	int nloc = get_local_size(0);

	T *Aprivate = malloc(sizeof(T)*rowDim);
	T temp;

	//Row major Format
	for(k=0;k<common;k++){
		Aprivate[k] = A[i*colDim+k];
	}

	for(j=0; j<colDim;j++){
		for(k=iloc;k<common;k+=nloc){
			localMem[k]=B[k*common+j];
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	tmp=0.0f;

	for(k=0; k<common;k++){
		tmp+=Aprivate[k]*BlocalMem[k];
	}
	C[i*colDim+j] += tmp;
}

