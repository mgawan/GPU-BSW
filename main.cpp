#include "kernel.hpp"
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <vector>

using namespace std;

int main()
{
    // READ SEQUENCE
  /*  string seqB =
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGTTTTTTC"
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCAAAAAAAAAAAAAAAAAAA";  // AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGTTTTTTTTT";
                                                      // // sequence A
    // CONTIG SEQUENCE
    string seqA =
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
        "AGAGAAGAGAGAGAGAAGAGAGAGAGGGG";
        */
vector<string> sequencesA, sequencesB;
//unsigned largestA = seqA.size(), largestB= seqB.size();

//////////////
//////////////

string myInLine;
ifstream ref_file("./test_data/ref_file.txt");
ifstream quer_file("./test_data/query_file.txt");
unsigned largestA = 0, largestB= 0;

if(ref_file.is_open())
{
while(getline(ref_file,myInLine))
{
string seq = myInLine.substr(myInLine.find(":")+1, myInLine.size()-1);
sequencesA.push_back(seq);
if(seq.size() > largestA){
  largestA = seq.size();
}
}
}

if(quer_file.is_open())
{
while(getline(quer_file,myInLine))
{
string seq = myInLine.substr(myInLine.find(":")+1, myInLine.size()-1);
sequencesB.push_back(seq);
if(seq.size() > largestB){
  largestB = seq.size();
}
}
}
cout <<"largestA:"<<largestA<<" largestB:"<<largestB<<endl;
//////////////
//////////////


   //   for(int i = 0; i < NBLOCKS; i++)
   // {
   //       sequencesA.push_back(seqA);
   //      sequencesB.push_back(seqB);
   //   }

//cout << "totalA:"<<sequencesA.size()<<" sizeA:"<<sequencesA[0].length()<<endl;
//cout << "totalB:"<<sequencesB.size()<<" sizeB:"<<sequencesB[0].length()<<endl;

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

    //	unsigned *offsetA = new unsignedf[nAseq*sizeof(int)];
    //	unsigned *offsetB = new unsigned[nBseq*sizeof(int)];
    for(int i = 0; i < sequencesA.size(); i++)
    {
        offsetA[i] = sequencesA[i].size();
    }

    for(int i = 0; i < sequencesB.size(); i++)
    {
        offsetB[i] = sequencesB[i].size();
    }


    auto start    = NOW;
    vec_offsetA_d = offsetA;
    vec_offsetB_d = offsetB;
  //  cout << "*******here here1" << endl;
    thrust::inclusive_scan(vec_offsetA_d.begin(), vec_offsetA_d.end(),
                           vec_offsetA_d.begin());
    thrust::inclusive_scan(vec_offsetB_d.begin(), vec_offsetB_d.end(),
                           vec_offsetB_d.begin());
//cout <<"prefix_sum calculated .."<<endl;
    unsigned totalLengthA = vec_offsetA_d[sequencesA.size() - 1];
    unsigned totalLengthB = vec_offsetB_d[sequencesB.size() - 1];

//cout << "totCharsA:"<<totalLengthA<<endl;
//cout << "totCharsB:"<<totalLengthB<<endl;
    unsigned* offsetA_d = thrust::raw_pointer_cast(&vec_offsetA_d[0]);
    unsigned* offsetB_d = thrust::raw_pointer_cast(&vec_offsetB_d[0]);

    unsigned offsetSumA = 0;
    unsigned offsetSumB = 0;

    vec_offsetMatrix_d = offsetMatrix;
    thrust::inclusive_scan(vec_offsetMatrix_d.begin(), vec_offsetMatrix_d.end(),
                           vec_offsetMatrix_d.begin());

//cout <<"dp matrix prefix_sum computed."<<endl;
    char* strA = new char[totalLengthA];
    char* strB = new char[totalLengthB];
    for(int i = 0; i < sequencesA.size(); i++)
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

  //  cout << "dpcells:" << endl << dp_matrices_cells << endl;

    cudaErrchk(cudaMalloc(&I_i, dp_matrices_cells * sizeof(short)));
    cudaErrchk(cudaMalloc(&I_j, dp_matrices_cells * sizeof(short)));

    // copy back
    cudaErrchk(cudaMalloc(&alAbeg_d, NBLOCKS * sizeof(short)));
    cudaErrchk(cudaMalloc(&alBbeg_d, NBLOCKS * sizeof(short)));
    cudaErrchk(cudaMalloc(&alAend_d, NBLOCKS * sizeof(short)));
    cudaErrchk(cudaMalloc(&alBend_d, NBLOCKS * sizeof(short)));

    //	for(int iter = 0; iter < 10; iter++){

    cudaErrchk(
        cudaMemcpy(strA_d, strA, totalLengthA * sizeof(char), cudaMemcpyHostToDevice));
    cudaErrchk(
        cudaMemcpy(strB_d, strB, totalLengthB * sizeof(char), cudaMemcpyHostToDevice));



    unsigned totShmem = 3 * 3 * (largestB + 1) * sizeof(short) + 3 * largestB  + (largestB  & 1) + largestA;
  //  cout << "shmem:" << totShmem << endl;
    unsigned alignmentPad = 4+(4 - totShmem % 4);
  //  cout << "alignmentpad:" << alignmentPad << endl;
  //  cout << "totShmem:"<<totShmem + alignmentPad +
      //  sizeof(int) * (sequencesA[0].size() + sequencesB[0].size() + 2)<<endl;
    align_sequences_gpu<<<NBLOCKS, largestB,
                          totShmem + alignmentPad +
                              sizeof(int) * (largestA + largestB+2)>>>(
        strA_d, strB_d, offsetA_d, offsetB_d, offsetMatrix_d, I_i, I_j, alAbeg_d,
        alAend_d, alBbeg_d, alBend_d);

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

    cudaErrchk(cudaFree(alAbeg_d));
    cudaErrchk(cudaFree(alBbeg_d));
    cudaErrchk(cudaFree(alAend_d));
    cudaErrchk(cudaFree(alBend_d));
// int error = 0;
//     for(int i = 0; i < NBLOCKS; i++){
//       if(alAbeg[i] != 189 || alAend[i] != 314 || alBbeg[i] != 1 || alBend[i] != 126){
//        cout << "i:"<<i<<" startA=" << alAbeg[i] << ", endA=" << alAend[i]<<" startB=" << alBbeg[i] << ", endB=" << alBend[i]<<endl;
//         error++;
//       }
//     }
//     cout <<"total errors:"<<error<<endl;

      //  cout << " startA=" << alAbeg[0] << ", endA=" << alAend[0]<<" startB=" << alBbeg[0] << ", endB=" << alBend[0]<<endl;
    return 0;
}
