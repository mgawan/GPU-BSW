#include "kernel.hpp"
#include <cmath>
#include <iostream>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <vector>

using namespace std;

int
main()
{
    // READ SEQUENCE
    string seqB = /*"GGGAAAAAAAGGGG";*/
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGTTTTTTC"
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCAAAAAAAAAAAA";  // AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGTTTTTTTTT";
                                                      // // sequence A
    // CONTIG SEQUENCE
    string seqA = /*"AAAAAAA";*/
        "GAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGA"
        "GAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGA"
        "GAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGGGGGGGGGGG"
        "GGGGGGGGGGGGGGGGGGGGGGGGGTTTTTTCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCAAAAAAAAAAAAAAAAGAG"
        "AGAGAGAAGAGAGAGAGAAGAGAGAGAGAAGAGAGAGAGAAGAGAGAGAGAAGAGAGAGAGAAGAGAGAGAGAAGAGAGA"
        "GAGAAGAGAGAGAGAAGAGAGAGAGAAGAGAGAGAGAAGAGAGAGAGAAGAGAGAGAGAAGAGAGAGAGAAGAGAGAGAG"
        "AAGAGAGAGAGAAGAGAGAGAGAAGAGAGAGAGAAGAGAGAGAGAAGAGAGAGAGAAGAGAGAGAGAAGAGAGAGAGAAG"
        "AGAGAGAGAAGAGAGAGAGAAGAGAGAGAGAAGAGAGAGAGAAGAGAGAGAGAAGAGAGAGAGAAGAGAGAGAGAAGAGA"
        "GAGAGAAGAGAGAGAGAAGAGAGAGAGAAGAGAGAGAGAAGAGAGAGAGAAGAGAGAGAGAAGAGAGAGAGAAAGAGAGA"
        "GAGAAGAGAGAGAGAAGAGAGAGAGAAGAGAGAGAGAAGAGAGAGAGAAGAGAGAGAGAAGAGAGAGAGAAGAGAGAGAG"
        "AAGAGAGAGAGAAGAGAGAGAGAAGAGAGAGAGAAGAGAGAGAGAAGAGAGAGAGAAGAGAGAGAGAAGAGAGAGAGAAG"
        "AGAGAGAGAAGAGAGAGAGAAGAGAGAGAGAAAGAGAGAGAGAAGAGAGAGAGAAGAGAGAGAGAAGAGAGAGAGAAGAG"
        "AGAGAGAAGAGAGAGAGAAGAGAGAGAGAAAGAGAGAGAGAAGAGAGAGAGAAGAGAGAGAGAAGAGAGAGAGAAGAGAG"
        "AGAGAAGAGAGAGAGAAGAGAGAGAGGGG";  // GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAATTTTTTTTTTTTTTTTTTTTTTTTTT";
                                          // // sequence B

    cout << "lengthA:" << seqA.size() << " lengthB:" << seqB.size() << endl;
    cout << seqA<<endl;
    vector<string> sequencesA, sequencesB;

    for(int i = 0; i < NBLOCKS; i++)
    {
        sequencesA.push_back(seqA);
        sequencesB.push_back(seqB);
    }
    //
    unsigned nAseq = NBLOCKS;
    unsigned nBseq = NBLOCKS;

    thrust::host_vector<int>        offsetA(sequencesA.size());
    thrust::host_vector<int>        offsetB(sequencesB.size());
    thrust::device_vector<unsigned> vec_offsetA_d(sequencesA.size());
    thrust::device_vector<unsigned> vec_offsetB_d(sequencesB.size());

    thrust::host_vector<unsigned>   offsetMatrix(NBLOCKS);        //*sizeof(unsigned));
    thrust::device_vector<unsigned> vec_offsetMatrix_d(NBLOCKS);  //*sizeof(unsigned));
    // offsetMatrix = (int*)malloc(NBLOCKS*sizeof(int));
    for(int i = 0; i < NBLOCKS; i++)
    {
        offsetMatrix[i] = (sequencesA[i].size() + 1) * (sequencesB[i].size() + 1);
    }

    //	unsigned *offsetA = new unsigned[nAseq*sizeof(int)];
    //	unsigned *offsetB = new unsigned[nBseq*sizeof(int)];
    for(int i = 0; i < nAseq; i++)
    {
        offsetA[i] = sequencesA[i].size();
    }

    for(int i = 0; i < nBseq; i++)
    {
        offsetB[i] = sequencesB[i].size();
    }

    /*	offsetA[0]=sequencesA[0].size();
      for(int i = 1; i < nAseq; i++){
          offsetA[i]=offsetA[i-1]+sequencesA[i].size();
      }

      offsetB[0]=sequencesB[0].size();
      for(int i = 1; i < nBseq; i++){
          offsetB[i]=offsetB[i-1]+sequencesB[i].size();
      }
  */
    auto start    = NOW;
    vec_offsetA_d = offsetA;
    vec_offsetB_d = offsetB;
    cout << "*******here here1" << endl;
    thrust::inclusive_scan(vec_offsetA_d.begin(), vec_offsetA_d.end(),
                           vec_offsetA_d.begin());
    thrust::inclusive_scan(vec_offsetB_d.begin(), vec_offsetB_d.end(),
                           vec_offsetB_d.begin());

    unsigned totalLengthA = vec_offsetA_d[nAseq - 1];
    unsigned totalLengthB = vec_offsetB_d[nBseq - 1];

    cout << "lengthA:" << totalLengthA << endl;
    cout << "lengthB:" << totalLengthB << endl;
    //  	unsigned totalLengthA = offsetA[nAseq-1];
    //	unsigned totalLengthB = offsetB[nBseq-1];
    unsigned* offsetA_d = thrust::raw_pointer_cast(&vec_offsetA_d[0]);
    unsigned* offsetB_d = thrust::raw_pointer_cast(&vec_offsetB_d[0]);
    // declare A and B strings
    // char* strA, *strB;
    // allocate and copy A string
    //	strA = (char*)malloc(sizeof(char)*totalLengthA);
    // strB = (char*)malloc(sizeof(char)*totalLengthB);

    // char *strA = new char[totalLengthA];
    // char *strB = new char[totalLengthB];
    unsigned offsetSumA = 0;
    unsigned offsetSumB = 0;
    cout << "*******here here2" << endl;

    vec_offsetMatrix_d = offsetMatrix;

    thrust::inclusive_scan(vec_offsetMatrix_d.begin(), vec_offsetMatrix_d.end(),
                           vec_offsetMatrix_d.begin());

    // char* strA, *strB;
    // allocate and copy A string
    // strA = (char*)malloc(sizeof(char)*totalLengthA);
    //	strB = (char*)malloc(sizeof(char)*totalLengthB);
    char* strA = new char[totalLengthA];
    char* strB = new char[totalLengthB];
    for(int i = 0; i < nAseq; i++)
    {
        char* seqptrA = strA + offsetSumA;  // vec_offsetA_d[i] - sequencesA[i].size();
        memcpy(seqptrA, sequencesA[i].c_str(), sequencesA[i].size());

        char* seqptrB = strB + offsetSumB;  // vec_offsetB_d[i] - sequencesB[i].size();
        memcpy(seqptrB, sequencesB[i].c_str(), sequencesB[i].size());
        offsetSumA += sequencesA[i].size();
        offsetSumB += sequencesB[i].size();
    }

    long      dp_matrices_cells = vec_offsetMatrix_d[NBLOCKS - 1];
    unsigned* offsetMatrix_d    = thrust::raw_pointer_cast(&vec_offsetMatrix_d[0]);

    char * strA_d, *strB_d;
    short *I_i, *I_j;  // device pointers for traceback matrices
                       // double *matrix, *Ematrix, *Fmatrix;
    short  alAbeg[NBLOCKS], alBbeg[NBLOCKS], alAend[NBLOCKS], alBend[NBLOCKS];
    short *alAbeg_d, *alBbeg_d, *alAend_d, *alBend_d;

    // cout << "allocating memory" << endl;

    cudaErrchk(cudaMalloc(&strA_d, totalLengthA * sizeof(char)));
    cudaErrchk(cudaMalloc(&strB_d, totalLengthB * sizeof(char)));

    cout << "dpcells:" << endl << dp_matrices_cells << endl;

    cudaErrchk(cudaMalloc(&I_i, dp_matrices_cells * sizeof(short)));
    cudaErrchk(cudaMalloc(&I_j, dp_matrices_cells * sizeof(short)));
    // cout << "allocating matrices" << endl;

    // copy back
    cudaErrchk(cudaMalloc(&alAbeg_d, NBLOCKS * sizeof(short)));
    cudaErrchk(cudaMalloc(&alBbeg_d, NBLOCKS * sizeof(short)));
    cudaErrchk(cudaMalloc(&alAend_d, NBLOCKS * sizeof(short)));
    cudaErrchk(cudaMalloc(&alBend_d, NBLOCKS * sizeof(short)));

    //	for(int iter = 0; iter < 10; iter++){
    cout << "*******here here6" << endl;
    cudaErrchk(
        cudaMemcpy(strA_d, strA, totalLengthA * sizeof(char), cudaMemcpyHostToDevice));
    cudaErrchk(
        cudaMemcpy(strB_d, strB, totalLengthB * sizeof(char), cudaMemcpyHostToDevice));

    // cudaProfilerStart();
    cout << "*******here here4" << endl;

    unsigned totShmem = 3 * 3 * (seqB.size() + 1) * sizeof(short) + 3 * seqB.size() +
                        (seqB.size() & 1) + seqA.size();
    cout << "shmem:" << totShmem << endl;
    unsigned alignmentPad = 4 - totShmem % 4;
    cout << "alignmentpad:" << alignmentPad << endl;
    align_sequences_gpu<<<NBLOCKS, seqB.size(),
                          totShmem + alignmentPad +
                              sizeof(int) * (seqA.size() + seqB.size() + 2)>>>(
        strA_d, strB_d, offsetA_d, offsetB_d, offsetMatrix_d, I_i, I_j, alAbeg_d,
        alAend_d, alBbeg_d, alBend_d);
    cout << "*******here here5" << endl;
    // cout << "kernel launched" << endl;
    // cudaProfilerStop();
    // cudaDeviceSynchronize();
    cudaErrchk(
        cudaMemcpy(alAbeg, alAbeg_d, NBLOCKS * sizeof(short), cudaMemcpyDeviceToHost));
    cudaErrchk(
        cudaMemcpy(alBbeg, alBbeg_d, NBLOCKS * sizeof(short), cudaMemcpyDeviceToHost));
    cudaErrchk(
        cudaMemcpy(alAend, alAend_d, NBLOCKS * sizeof(short), cudaMemcpyDeviceToHost));
    cudaErrchk(
        cudaMemcpy(alBend, alBend_d, NBLOCKS * sizeof(short), cudaMemcpyDeviceToHost));

    //}
    auto                     end  = NOW;
    chrono::duration<double> diff = end - start;
    cout << "time = " << diff.count() << endl;
    cudaErrchk(cudaFree(strA_d));
    cudaErrchk(cudaFree(strB_d));
    cudaErrchk(cudaFree(I_i));
    cudaErrchk(cudaFree(I_j));
    //	cudaErrchk(cudaFree(offsetA_d));
    //	cudaErrchk(cudaFree(offsetB_d));
    //	cudaErrchk(cudaFree(offsetMatrix_d));

    cudaErrchk(cudaFree(alAbeg_d));
    cudaErrchk(cudaFree(alBbeg_d));
    cudaErrchk(cudaFree(alAend_d));
    cudaErrchk(cudaFree(alBend_d));

    cout << "startA=" << alAbeg[0] << ", endA=" << alAend[0]
         << " start2A=" << alAbeg[9000] << " end2A=" << alAend[9000] << endl;
    cout << "startB=" << alBbeg[0] << ", endB=" << alBend[0]
         << " start2B=" << alBbeg[9000] << " end2B=" << alBend[9000] << endl;

    return 0;
}
