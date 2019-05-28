#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <sys/time.h>
#include <chrono>
using namespace std;

#define EXTEND_GAP -1
#define START_GAP -2
#define NBLOCKS 31000
#define NOW std::chrono::high_resolution_clock::now()

#define cudaErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true){

	if(code != cudaSuccess){
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	if(abort) exit(code);
	}
}
//int ind;

//double similarityScore(char a, char b);
//double findMax(double array[], int length)

__inline__ __device__
short warpReduceMax(short val, short *myIndex, short *myIndex2) {
int warpSize = 32;
short myMax = 0;
short newInd = 0;
short newInd2 = 0;
short ind = *myIndex;
short ind2 = *myIndex2;


  myMax = val;
  unsigned mask = __ballot_sync(0xffffffff, threadIdx.x < blockDim.x);

  for (int offset = warpSize/2; offset > 0; offset /= 2){

    val = max(val,__shfl_down_sync(mask, val, offset));
    newInd = __shfl_down_sync(mask, ind, offset);
    newInd2 = __shfl_down_sync(mask, ind2, offset);

    if(val != myMax){

    	ind = newInd;
    	ind2 = newInd2;
    	myMax = val;

    	}
    }

		*myIndex = ind;
		*myIndex2 = ind2;
		val = myMax;
  return val;
}


__device__
short blockShuffleReduce(short myVal, short *myIndex, short *myIndex2){
	int laneId = threadIdx.x % 32;
	int warpId = threadIdx.x / 32;
	__shared__ int locTots[32];
	__shared__ int locInds[32];
	short myInd = *myIndex;
	short myInd2 = *myIndex2;
	myVal = warpReduceMax(myVal, &myInd, &myInd2);


	if(laneId == 0) locTots[warpId] = myVal;
	if(laneId == 0) locInds[warpId] = myInd;

	__syncthreads();

	if(threadIdx.x <= (blockDim.x / 32)){
		myVal = locTots[threadIdx.x];
		myInd = locInds[threadIdx.x];
	}else{
		myVal = 0;
		myInd = -1;
		myInd2 = -1;
	}


	if(warpId == 0) myVal = warpReduceMax(myVal, &myInd, &myInd2);
	*myIndex = myInd;
	*myIndex2 = myInd2;

return myVal;
}



__device__ __host__
double similarityScore(char a, char b)
{
	double result;
	if(a==b)
	{
		result=5;
	}
	else
	{
		result=-3;
	}
	return result;
}

__device__ __host__
 short findMax(short array[], int length, int *ind)
{
	short max = array[0];
	*ind = 0;

	for(int i=1; i<length; i++)
	{
		if(array[i] > max)
		{
			max = array[i];
			*ind=i;
		}
	}
	return max;
}

__device__ __noinline__
void traceBack(short current_i, short current_j, short* seqA_align_begin, short* seqB_align_begin,
const char* seqA, const char* seqB,short* I_i, short* I_j, unsigned lengthSeqB){
      // int current_i=i_max,current_j=j_max;
	int myId = blockIdx.x;
        short next_i=I_i[current_i*(lengthSeqB+1) + current_j];
        short next_j=I_j[current_i*(lengthSeqB+1) + current_j];


while(((current_i!=next_i) || (current_j!=next_j)) && (next_j!=0) && (next_i!=0))
        {



                current_i = next_i;
                current_j = next_j;
                next_i = I_i[current_i*(lengthSeqB+1) + current_j];
                next_j = I_j[current_i*(lengthSeqB+1) + current_j];

        }
	//printf("final current_i=%d, current_j=%d\n", current_i, current_j);
	seqA_align_begin[myId] = current_i;
	seqB_align_begin[myId] = current_j;
	//printf("traceback done\n");

       // *tick_out = tick;
}


__global__
void align_sequences_gpu(char* seqA_array,char* seqB_array, unsigned* prefix_lengthA,
unsigned* prefix_lengthB, unsigned* prefix_matrices, short *I_i_array, short *I_j_array,
short* seqA_align_begin, short* seqA_align_end, short* seqB_align_begin, short* seqB_align_end){

	int myId = blockIdx.x;
	int myTId = threadIdx.x;

	unsigned lengthSeqA;
	unsigned lengthSeqB;
	unsigned matrixOffset;
	//local pointers
	char* seqA;
	char* seqB;
	short* I_i, *I_j;

	extern __shared__ char is_valid_array[];
	char* is_valid = &is_valid_array[0];

	if(myId == 0){
		lengthSeqA = prefix_lengthA[0];
		lengthSeqB = prefix_lengthB[0];
		matrixOffset = 0;
		seqA = seqA_array;
		seqB = seqB_array;
		I_i = I_i_array;
		I_j = I_j_array;
	}else{
		lengthSeqA = prefix_lengthA[myId] - prefix_lengthA[myId-1];
		lengthSeqB = prefix_lengthB[myId] - prefix_lengthB[myId-1];
		matrixOffset = prefix_matrices[myId-1];
		seqA = seqA_array + prefix_lengthA[myId-1];
		seqB = seqB_array + prefix_lengthB[myId-1];
		I_i = I_i_array + matrixOffset;
		I_j = I_j_array + matrixOffset;
	}

	short* curr_H = (short*)(&is_valid_array[3*lengthSeqB+(lengthSeqB&1)]);// point where the valid_array ends
	short* prev_H = &curr_H[lengthSeqB+1]; // where the curr_H array ends
	short* prev_prev_H = &prev_H[lengthSeqB+1];

	short* curr_E = &prev_prev_H[lengthSeqB+1];
	short* prev_E = &curr_E[lengthSeqB+1];
	short* prev_prev_E = &prev_E[lengthSeqB+1];

	short* curr_F = &prev_prev_E[lengthSeqB+1];
	short* prev_F = &curr_F[lengthSeqB+1];
	short* prev_prev_F = &prev_F[lengthSeqB+1];

	//char* v = is_valid;

	memset(is_valid, 0, lengthSeqB);
	//if(myId == 0) printf("is_valid= %lu\n", is_valid);
	is_valid += lengthSeqB;
	//if(myId == 0) printf("is_valid= %lu\n", is_valid);
	memset(is_valid, 1, lengthSeqB);
	is_valid += lengthSeqB;
	//if(myId == 0) printf("is_valid= %lu\n", is_valid);
	memset(is_valid, 0, lengthSeqB);
	//if(myId == 0) for(int k = 0; k < 3*lengthSeqB; k++) printf("%d ", f[k]);




	if(myTId == 0) {
		memset(curr_H, 0, 9*(lengthSeqB+1)*sizeof(short));
	}
	//__shared__ int global_max;
	__shared__ int i_max;
	__shared__ int j_max;

	__syncthreads();



	short traceback[4];

	int ind;

	int i = 1;
	short thread_max = 0;
	short thread_max_i = 0;
	short thread_max_j = 0;

	short* tmp_ptr;

	for(int diag = 0; diag < lengthSeqA + lengthSeqB - 1; diag++){ // iterate for the number of anti-diagonals

		int j = myTId+1;

		is_valid = is_valid - (diag < lengthSeqB || diag >= lengthSeqA);

		tmp_ptr = prev_H;
		prev_H = curr_H;
		curr_H = prev_prev_H;
		prev_prev_H = tmp_ptr;

		memset(curr_H, 0, (lengthSeqB+1)*sizeof(short));

		tmp_ptr = prev_E;
		prev_E = curr_E;
		curr_E = prev_prev_E;
		prev_prev_E = tmp_ptr;

		memset(curr_F, 0, (lengthSeqB+1)*sizeof(short));

		tmp_ptr = prev_F;
		prev_F = curr_F;
		curr_F = prev_prev_F;
		prev_prev_F = tmp_ptr;

		memset(curr_E, 0, (lengthSeqB+1)*sizeof(short));

		//if(myId == 0 ) printf("j=%d, i=%d, is_valid=%d,cond=%d\n", j, i,is_valid[myTId],(diag < lengthSeqB || diag >= lengthSeqA));
		__syncthreads();

		if(is_valid[myTId]){
			short fVal = prev_F[j]+EXTEND_GAP;
			short hfVal = prev_H[j]+START_GAP;
			short eVal = prev_E[j-1]+EXTEND_GAP;
			short heVal = prev_H[j-1]+START_GAP;

			curr_F[j] = (fVal > hfVal) ? fVal : hfVal;
			curr_E[j] = (eVal > heVal) ? eVal: heVal;



			//curr_F[j] = ((prev_F[j]+EXTEND_GAP)>(prev_H[j]+START_GAP)) ? prev_F[j]+EXTEND_GAP : prev_H[j]+START_GAP;
		//	curr_E[j] = ((prev_E[j-1]+EXTEND_GAP)>(prev_H[j-1]+START_GAP)) ? prev_E[j-1]+EXTEND_GAP : prev_H[j-1]+START_GAP;



			traceback[0] = prev_prev_H[j-1]+similarityScore(seqA[i-1],seqB[j-1]);
			traceback[1] = curr_F[j];
			traceback[2] = curr_E[j];
			traceback[3] = 0;

		/*	curr_H[j] =  prev_prev_H[j-1]+similarityScore(seqA[i-1],seqB[j-1]);
			curr_H[j] =  curr_H[j] >= curr_F[j]? curr_H[j]: curr_F[j];
			curr_H[j] =  curr_H[j] >= curr_E[j]? curr_H[j]: curr_E[j];
			curr_H[j] =  curr_H[j] >= 0? curr_H[j]: 0;
			*/

			curr_H[j] = findMax(traceback,4,&ind);



			thread_max_i = (thread_max >= curr_H[j]) ? thread_max_i : i;
			thread_max_j = (thread_max >= curr_H[j]) ? thread_max_i : myTId+1;
		thread_max = thread_max >= curr_H[j] ? thread_max : curr_H[j];

			//if(myId==0) printf("id=%d thread_max=%d, i=%d\n", myTId, thread_max, thread_max_i);

			switch(ind)
			{
				case 0:
					I_i[i*(lengthSeqB+1)+j] = i-1;
					I_j[i*(lengthSeqB+1)+j] = j-1;
					break;
				case 1:
					I_i[i*(lengthSeqB+1)+j] = i-1;
                    			I_j[i*(lengthSeqB+1)+j] = j;
                    			break;
				case 2:
					I_i[i*(lengthSeqB+1)+j] = i;
                    			I_j[i*(lengthSeqB+1)+j] = j-1;
                    			break;
				case 3:
					I_i[i*(lengthSeqB+1)+j] = i;
                    			I_j[i*(lengthSeqB+1)+j] = j;
                    			break;
			}
			i++;
        }
	}

	//atomicMax(&global_max, thread_max);
	thread_max = blockShuffleReduce(thread_max, &thread_max_i, &thread_max_j);// thread 0 will have the correct values

	// traceback
	if(myTId == 0) {
		i_max = thread_max_i;
		j_max = thread_max_j;
		int current_i=i_max,current_j=j_max;
		seqA_align_end[myId] = current_i;
		seqB_align_end[myId] = current_j;

		traceBack(current_i, current_j, seqA_align_begin, seqB_align_begin, seqA, seqB, I_i, I_j, lengthSeqB);

	}
}

__host__
void traceBack_cpu(int current_i, int current_j, int* seqA_align_begin, int*seqB_align_begin, const char* seqA, const char* seqB,int lengthSeqA, int** I_i, int** I_j){
      // int current_i=i_max,current_j=j_max;
      //  int next_i=I_i[current_i][current_j];
       // int next_j=I_j[current_i][current_j];
        //int tick=0;
	cout << "lengthSeqA=" << lengthSeqA << endl;
        int next_i = I_i[current_i][current_j];
        int next_j = I_j[current_i][current_j];
	cout << "curr_i=" << current_i << ", curr_j=" << current_j << ", next_i=" << next_i << ", next_j=" << next_j << endl;
	printf("starting traceback\n");
        while(((current_i!=next_i) || (current_j!=next_j)) && (next_j!=0) && (next_i!=0))
        {


                current_i = next_i;
                current_j = next_j;
		cout << "curr_i=" << current_i << ", curr_j=" << current_j << endl;
                next_i = I_i[current_i][current_j];
                next_j = I_j[current_i][current_j];
                //*tick++;
        }
        *seqA_align_begin = current_i;
        *seqB_align_begin = current_j;
	//printf("traceback done\n");
  //return tick;
       // *tick_out = tick;
}

void align_sequences_cpu(const char* seqA, const char* seqB, int* seqA_align_begin, int* seqA_align_end, int* seqB_align_begin, int* seqB_align_end, int lengthA, int lengthB){


	// initialize some variables
	int lengthSeqA = lengthA;
	int lengthSeqB = lengthB;

	// initialize matrix
	double matrix[lengthSeqA+1][lengthSeqB+1];
	double Fmatrix[lengthSeqA+1][lengthSeqB+1];
	double Ematrix[lengthSeqA+1][lengthSeqB+1];

	for(int i=0;i<=lengthSeqA;i++)
	{
		for(int j=0;j<=lengthSeqB;j++)
		{
			matrix[i][j]=0;
			Fmatrix[i][j]=0;
			Ematrix[i][j]=0;
		}
	}

	short traceback[4];
	int **I_i;
	int **I_j;
	I_i = new int *[lengthSeqA+1];
	I_j = new int *[lengthSeqA+1];
	for(int i = 0; i < lengthSeqA+1; i++){
		I_i[i] = new int[lengthSeqB+1];
		I_j[i] = new int[lengthSeqB+1];
	}
	int ind;
	//start populating matrix
	for (int i=1;i<=lengthSeqA;i++)
	{
		for(int j=1;j<=lengthSeqB;j++)
        {                       // cout << i << " " << j << endl;

			Fmatrix[i][j] = ((Fmatrix[i-1][j]+EXTEND_GAP)>(matrix[i-1][j]+START_GAP)) ? Fmatrix[i-1][j]+EXTEND_GAP:matrix[i-1][j]+START_GAP;
			Ematrix[i][j] = ((Ematrix[i][j-1]+EXTEND_GAP)>(matrix[i][j-1]+START_GAP)) ? Ematrix[i][j-1]+EXTEND_GAP:matrix[i][j-1]+START_GAP;

			traceback[0] = matrix[i-1][j-1]+similarityScore(seqA[i-1],seqB[j-1]);
			traceback[1] = Fmatrix[i][j];
			traceback[2] = Ematrix[i][j];
			traceback[3] = 0;

			matrix[i][j] = findMax(traceback,4,&ind);
			switch(ind)
			{
				case 0:
					I_i[i][j] = i-1;
					I_j[i][j] = j-1;
					break;
				case 1:
					I_i[i][j] = i-1;
					I_j[i][j] = j;
                    			break;
				case 2:
					I_i[i][j] = i;
					I_j[i][j] = j-1;
                    			break;
				case 3:
					I_i[i][j] = i;
					I_j[i][j] = j;
                    			break;
			}
        }
	}


	// find the max score in the matrix
	cout << "lA =" << lengthSeqA << ", l B = " << lengthSeqB << endl;
	double matrix_max = 0;
	int i_max=0, j_max=0;
	for(int i=1;i<lengthSeqA;i++)
	{
		for(int j=1;j<lengthSeqB;j++)
		{
			if(matrix[i][j]>matrix_max)
			{
				matrix_max = matrix[i][j];
				i_max=i;
				j_max=j;
			}
		}
	}
	cout << "i_max =" << i_max << ", j_max=" << j_max << endl;

//	cout << "Max score in the matrix is " << matrix_max << endl;

	// traceback

	int current_i=i_max,current_j=j_max;
	int next_i=I_i[current_i][current_j];
	int next_j=I_j[current_i][current_j];
	//int tick=0;
	cout << "next_i_out=" << next_i << ", next_j_out=" << next_j << endl;
	*seqA_align_end = current_i;
	*seqB_align_end = current_j;
	traceBack_cpu(current_i, current_j, seqA_align_begin, seqB_align_begin, seqA, seqB, lengthSeqA, I_i, I_j);

}

int main()
{

	//READ SEQUENCE
	string seqB = /*"GGGAAAAAAAGGGG";*/"CTTGTAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGTTTTTTCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCAAAAAAAAAAAAAAA";//AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGTTTTTTTTT"; // sequence A
	//CONTIG SEQUENCE
	string seqA = /*"AAAAAAA";*/"CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAATTTTTTTTTTTTTTTTTTTTTTTTTTTTCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAATTTTTTTTTTTTTTTTTTTTTTTTTTCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAATTTTTTTTTTTTTTTTTTTTTTTTTTCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAATTTTTTTTTTTTTTTTTTTTTTTTTTCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG";//GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAATTTTTTTTTTTTTTTTTTTTTTTTTT"; // sequence B

  	vector<string> sequencesA, sequencesB;

	for(int i = 0; i < NBLOCKS; i++) {
  		sequencesA.push_back(seqA);
  		sequencesB.push_back(seqB);
	}
  	//
	unsigned nAseq = NBLOCKS;
	unsigned nBseq = NBLOCKS;

  	unsigned *offsetA, *offsetB;
  	offsetA = (unsigned*)malloc(nAseq*sizeof(int));
  	offsetB = (unsigned*)malloc(nBseq*sizeof(int));

  	offsetA[0]=sequencesA[0].size();
  	for(int i = 1; i < nAseq; i++){
  		offsetA[i]=offsetA[i-1]+sequencesA[i].size();
  	}

  	offsetB[0]=sequencesB[0].size();
  	for(int i = 1; i < nBseq; i++){
  		offsetB[i]=offsetB[i-1]+sequencesB[i].size();
  	}

  	unsigned totalLengthA = offsetA[nAseq-1];
  	unsigned totalLengthB = offsetB[nBseq-1];

  	//declare A and B strings
	char* strA, *strB;
  	//allocate and copy A string
  	strA = (char*)malloc(sizeof(char)*totalLengthA);
  	strB = (char*)malloc(sizeof(char)*totalLengthB);
  	for(int i = 0; i<nAseq; i++){
  	 	char *seqptrA = strA + offsetA[i] - sequencesA[i].size();
  	 	memcpy(seqptrA, sequencesA[i].c_str(), sequencesA[i].size());

  	 	char *seqptrB = strB + offsetB[i] - sequencesB[i].size();
  	 	memcpy(seqptrB, sequencesB[i].c_str(), sequencesB[i].size());
  	}

  	//allocate and copy B string
  //	strB = (char*)malloc(sizeof(char)*totalLengthB);
  //	for(int i = 0; i<nBseq; i++){
  //	 	char *seqptr = strB + offsetB[i] - sequencesB[i].size();
  //	 	memcpy(seqptr, sequencesB[i].c_str(), sequencesB[i].size());
  //	}
	auto start = NOW;
  	unsigned *offsetMatrix;
  	offsetMatrix = (unsigned*)malloc(NBLOCKS*sizeof(int));
  	offsetMatrix[0]=(sequencesA[0].size()+1)*(sequencesB[0].size()+1); // offsets for traceback matrices
  	for(int i = 1; i < NBLOCKS; i++){
  		offsetMatrix[i]=offsetMatrix[i-1]+(sequencesA[i].size()+1)*(sequencesB[i].size()+1);
  	}

	//int lengthSeqB = seqB.size();

	char *strA_d, *strB_d;
	unsigned *offsetA_d, *offsetB_d;
	unsigned *offsetMatrix_d;
	short *I_i, *I_j; // device pointers for traceback matrices
	//double *matrix, *Ematrix, *Fmatrix;
  	short alAbeg[NBLOCKS], alBbeg[NBLOCKS], alAend[NBLOCKS], alBend[NBLOCKS];
  	short *alAbeg_d, *alBbeg_d, *alAend_d, *alBend_d;

	//cout << "allocating memory" << endl;

	cudaErrchk(cudaMalloc(&strA_d, totalLengthA*sizeof(char)));
	cudaErrchk(cudaMalloc(&strB_d, totalLengthB*sizeof(char)));

	//cout << "allocated strings"  << endl;
	//malloc matrices
//WHY ARW WE USING INTS FOR THIS??

	long dp_matrices_cells = 0;
	for(int i = 0; i < nAseq; i++) {
		dp_matrices_cells += (sequencesA[i].size()+1) * (sequencesB[i].size()+1);
	}
	//cout << dp_matrices_cells << endl;


	cudaErrchk(cudaMalloc(&I_i, dp_matrices_cells*sizeof(short)));
	cudaErrchk(cudaMalloc(&I_j, dp_matrices_cells*sizeof(short)));
	//cout << "allocating matrices" << endl;


// use character arrays for output data
	//allocating offsets
	cudaErrchk(cudaMalloc(&offsetA_d, nAseq*sizeof(int))); // array for storing offsets for A
	cudaErrchk(cudaMalloc(&offsetB_d, nBseq*sizeof(int)));  // array for storing offsets for B
	cudaErrchk(cudaMalloc(&offsetMatrix_d, NBLOCKS*sizeof(int))); // array for storing ofsets for traceback matrix

	// copy back
  	cudaErrchk(cudaMalloc(&alAbeg_d, NBLOCKS*sizeof(short)));
  	cudaErrchk(cudaMalloc(&alBbeg_d, NBLOCKS*sizeof(short)));
  	cudaErrchk(cudaMalloc(&alAend_d, NBLOCKS*sizeof(short)));
  	cudaErrchk(cudaMalloc(&alBend_d, NBLOCKS*sizeof(short)));

	for(int iter = 0; iter < 10; iter++){

	cudaErrchk(cudaMemcpy(strA_d, strA, totalLengthA*sizeof(char), cudaMemcpyHostToDevice));
	cudaErrchk(cudaMemcpy(strB_d, strB, totalLengthB*sizeof(char), cudaMemcpyHostToDevice));
	cudaErrchk(cudaMemcpy(offsetA_d, offsetA, nAseq*sizeof(int), cudaMemcpyHostToDevice));
	cudaErrchk(cudaMemcpy(offsetB_d, offsetB, nBseq*sizeof(int), cudaMemcpyHostToDevice));
	cudaErrchk(cudaMemcpy(offsetMatrix_d, offsetMatrix, NBLOCKS*sizeof(int), cudaMemcpyHostToDevice));


  	//cout << "launching kernel" << endl;
	align_sequences_gpu<<<NBLOCKS, seqB.size(), 3*3*(seqB.size()+1)*sizeof(short)+3*seqB.size()+(seqB.size()&1) >>>(strA_d, strB_d, offsetA_d, offsetB_d, offsetMatrix_d, I_i, I_j, alAbeg_d, alAend_d, alBbeg_d, alBend_d);

	//cout << "kernel launched" << endl;


	cudaErrchk(cudaMemcpy(alAbeg, alAbeg_d, NBLOCKS*sizeof(short), cudaMemcpyDeviceToHost));
	cudaErrchk(cudaMemcpy(alBbeg, alBbeg_d, NBLOCKS*sizeof(short), cudaMemcpyDeviceToHost));
	cudaErrchk(cudaMemcpy(alAend, alAend_d, NBLOCKS*sizeof(short), cudaMemcpyDeviceToHost));
	cudaErrchk(cudaMemcpy(alBend, alBend_d, NBLOCKS*sizeof(short), cudaMemcpyDeviceToHost));

	}
	auto end = NOW;
	chrono::duration<double> diff = end - start;
	cout << "time = " << diff.count() << endl;
	cudaErrchk(cudaFree(strA_d));
	cudaErrchk(cudaFree(strB_d));
	cudaErrchk(cudaFree(I_i));
	cudaErrchk(cudaFree(I_j));
	cudaErrchk(cudaFree(offsetA_d));
	cudaErrchk(cudaFree(offsetB_d));
	cudaErrchk(cudaFree(offsetMatrix_d));

	cudaErrchk(cudaFree(alAbeg_d));
	cudaErrchk(cudaFree(alBbeg_d));
	cudaErrchk(cudaFree(alAend_d));
	cudaErrchk(cudaFree(alBend_d));

	cout << "startA=" << alAbeg[8] << ", endA=" << alAend[8] << " start2A=" << alAbeg[29000] << " end2A=" << alAend[29000] << endl;
	cout << "startB=" << alBbeg[8] << ", endB=" << alBend[8] << " start2B=" << alBbeg[29000] << " end2B=" << alBend[29000] << endl;



	return 0;
}
