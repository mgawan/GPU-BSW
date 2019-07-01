#include "kernel.hpp"
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <vector>
#include<sstream>
#include<omp.h>

using namespace std;

int main(int argc, char *argv[])
{


int deviceCount;
cudaGetDeviceCount(&deviceCount);
cudaDeviceProp prop[deviceCount];
for(int i = 0; i < deviceCount; i++)
  cudaGetDeviceProperties(&prop[i], 0);

for(int i = 0; i < deviceCount; i++){
  cout <<"total Global Memory for Device "<<i<<":"<<prop[i].totalGlobalMem<<endl;
  cout <<"Compute version of device "<<i<<":"<<prop[i].major<<endl;
}


vector<string> G_sequencesA, G_sequencesB;// sequence A is the longer one/reference string


string myInLine;
ifstream ref_file(argv[1]);//"./test_data/ref_file_1.txt"
ifstream quer_file(argv[2]);//"./test_data/que_file_1.txt"
unsigned largestA = 0, largestB= 0;

int totSizeA = 0, totSizeB = 0;
if(ref_file.is_open())
{
while(getline(ref_file,myInLine))
{
string seq = myInLine.substr(myInLine.find(":")+1, myInLine.size()-1);
G_sequencesA.push_back(seq);
totSizeA += seq.size();
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
G_sequencesB.push_back(seq);
totSizeB += seq.size();
if(seq.size() > largestB){
  largestB = seq.size();
}
}
}
unsigned NBLOCKS = G_sequencesA.size();
cout <<"strings read:"<<NBLOCKS<<endl;
unsigned maxMatrixSize = (largestA+1)*(largestB+1);
long long totMemEst = largestA*G_sequencesA.size() + largestB*G_sequencesB.size() + maxMatrixSize*G_sequencesA.size()*sizeof(short)*2 + G_sequencesA.size()*sizeof(short)*4;

cout <<"totMemReq:"<<totMemEst<<endl;

// determining number of iterations required on a single GPUassert
int its = 0;
long long estMem = totMemEst;
while(estMem > (prop[0].totalGlobalMem*0.90)){
  its++;
  estMem /= its;
}



cout <<"largestA:"<<largestA<<" largestB:"<<largestB<<endl;

//#pragma omp parallel
//{
  int totThreads = omp_get_num_threads();
  cout <<"total threads:"<< totThreads<<endl;
  int my_cpu_id = omp_get_thread_num();
  cudaSetDevice(my_cpu_id);
  int myGPUid;
  cudaGetDevice(&myGPUid);

  cout <<" gpuid:"<<myGPUid<<" cpuID:"<<my_cpu_id<<endl;
  //vector<string> sequencesA, sequencesB;
  unsigned leftOvers = NBLOCKS%its;
  unsigned stringsPerIt = NBLOCKS/its;

  cout <<"Total iterations:"<<its<<endl;
  cout <<"Alignments Per Iteration:"<<stringsPerIt<<endl;
  cout <<"Lef over:"<<leftOvers<<endl;
    short *alAbeg = new short[NBLOCKS];
    short *alBbeg = new short[NBLOCKS];
    short *alAend = new short[NBLOCKS];
    short *alBend = new short[NBLOCKS]; // memory on CPU for copying the results

    short *test_Abeg = alAbeg;
    short *test_Bbeg = alBbeg;
    short *test_Aend = alAend;
    short *test_Bend = alBend;



  for(int perGPUIts = 0; perGPUIts < its; perGPUIts++){
    int blocksLaunched = 0;
    vector<string>::const_iterator beginAVec;
    vector<string>::const_iterator endAVec;
    vector<string>::const_iterator beginBVec;
    vector<string>::const_iterator endBVec;
    if(perGPUIts == its-1){
      beginAVec = G_sequencesA.begin()+(perGPUIts*stringsPerIt);
      endAVec = G_sequencesA.begin()+((perGPUIts+1)*stringsPerIt)+leftOvers; // so that each openmp thread has a copy of strings it needs to align
      beginBVec = G_sequencesB.begin()+(perGPUIts*stringsPerIt);
      endBVec = G_sequencesB.begin()+((perGPUIts+1)*stringsPerIt)+leftOvers; // so that each openmp thread has a copy of strings it needs to align

      blocksLaunched = stringsPerIt + leftOvers;
    }else{
      beginAVec = G_sequencesA.begin()+(perGPUIts*stringsPerIt);
      endAVec = G_sequencesA.begin()+(perGPUIts+1)*stringsPerIt; // so that each openmp thread has a copy of strings it needs to align
      beginBVec = G_sequencesB.begin()+(perGPUIts*stringsPerIt);
      endBVec = G_sequencesB.begin()+(perGPUIts+1)*stringsPerIt; // so that each openmp thread has a copy of strings it needs to align
          blocksLaunched = stringsPerIt;
    }

    vector<string> sequencesA(beginAVec,endAVec);
    vector<string> sequencesB(beginBVec, endBVec);

    cout <<"vecAsize:"<<sequencesA.size()<<endl;
    cout <<"vecBsize:"<<sequencesB.size()<<endl;

//  sequencesB = G_sequencesB;
    thrust::host_vector<int>        offsetA(sequencesA.size());
    thrust::host_vector<int>        offsetB(sequencesB.size());
    thrust::device_vector<unsigned> vec_offsetA_d(sequencesA.size());
    thrust::device_vector<unsigned> vec_offsetB_d(sequencesB.size());

    //thrust::host_vector<unsigned>   offsetMatrix(NBLOCKS);        //*sizeof(unsigned));
    //thrust::device_vector<unsigned> vec_offsetMatrix_d(NBLOCKS);  //*sizeof(unsigned));

    // for(int i = 0; i < NBLOCKS; i++)
    // {
    //     offsetMatrix[i] = (sequencesA[i].size() + 1) * (sequencesB[i].size() + 1);
    // }
cout <<"Asize:"<<sequencesA.size()<<endl;

    for(int i = 0; i < sequencesA.size(); i++)
    {
        offsetA[i] = sequencesA[i].size();
    }

    for(int i = 0; i < sequencesB.size(); i++)
    {
        offsetB[i] = sequencesB[i].size();
    }


  auto start    = NOW;
for(int i = 0; i < 1; i++){

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

  //  vec_offsetMatrix_d = offsetMatrix;
  //  thrust::inclusive_scan(vec_offsetMatrix_d.begin(), vec_offsetMatrix_d.end(),
  //                         vec_offsetMatrix_d.begin());


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

  //  long      dp_matrices_cells = vec_offsetMatrix_d[NBLOCKS - 1];
  //  unsigned* offsetMatrix_d    = thrust::raw_pointer_cast(&vec_offsetMatrix_d[0]);

    char * strA_d, *strB_d;
    short *I_i, *I_j;  // device pointers for traceback matrices
                       // double *matrix, *Ematrix, *Fmatrix;

    short *alAbeg_d, *alBbeg_d, *alAend_d, *alBend_d;
    //unsigned *offsetMatrix_d;
    // cout << "allocating memory" << endl;

    //cudaErrchk(cudaMalloc(&offsetMatrix_d, (largestA+1)*(largestB+1)*NBLOCKS* sizeof(unsigned)));
    cudaErrchk(cudaMalloc(&strA_d, totalLengthA * sizeof(char)));
    cudaErrchk(cudaMalloc(&strB_d, totalLengthB * sizeof(char)));

  //  cout << "dpcells:" << endl << dp_matrices_cells << endl;

  //  cudaErrchk(cudaMalloc(&I_i, dp_matrices_cells * sizeof(short)));
  //  cudaErrchk(cudaMalloc(&I_j, dp_matrices_cells * sizeof(short)));

  cudaErrchk(cudaMalloc(&I_i, maxMatrixSize*blocksLaunched* sizeof(short)));
  cudaErrchk(cudaMalloc(&I_j, maxMatrixSize*blocksLaunched* sizeof(short)));

    // copy back
    cudaErrchk(cudaMalloc(&alAbeg_d, blocksLaunched * sizeof(short)));
    cudaErrchk(cudaMalloc(&alBbeg_d, blocksLaunched * sizeof(short)));
    cudaErrchk(cudaMalloc(&alAend_d, blocksLaunched * sizeof(short)));
    cudaErrchk(cudaMalloc(&alBend_d, blocksLaunched * sizeof(short)));

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
    align_sequences_gpu<<<blocksLaunched, largestB,
                          totShmem + alignmentPad +
                              sizeof(int) * (largestA + largestB+2)>>>(
        strA_d, strB_d, offsetA_d, offsetB_d, maxMatrixSize, I_i, I_j, alAbeg_d,
        alAend_d, alBbeg_d, alBend_d);

        // offset pouinters for correct copy back


        cout <<"blocksLaunched:"<<blocksLaunched<<endl;

        cout <<"alAbeg:"<<alAbeg<<endl;

        size_t free, total;
        cudaMemGetInfo	(	&free, &total	);
        cout <<"free memory percent: "<<((float)free/total)<<" totalMem:"<<total<<endl;

    cudaErrchk(
        cudaMemcpy(alAbeg, alAbeg_d, blocksLaunched * sizeof(short), cudaMemcpyDeviceToHost));
        cout <<"this 1 copied back"<<endl;
    cudaErrchk(
        cudaMemcpy(alBbeg, alBbeg_d, blocksLaunched * sizeof(short), cudaMemcpyDeviceToHost));
          cout <<"this 2 copied back"<<endl;
    cudaErrchk(
        cudaMemcpy(alAend, alAend_d, blocksLaunched * sizeof(short), cudaMemcpyDeviceToHost));
    cudaErrchk(
        cudaMemcpy(alBend, alBend_d, blocksLaunched * sizeof(short), cudaMemcpyDeviceToHost)); // this does not cause the error the other three lines do.

    //}
    alAbeg += stringsPerIt;//perGPUIts;//*stringsPerIt;
    alBbeg += stringsPerIt;//;//*stringsPerIt;
    alAend += stringsPerIt;//;//*stringsPerIt;
    alBend += stringsPerIt;//;//*stringsPerIt;
    cudaErrchk(cudaFree(strA_d));
    cudaErrchk(cudaFree(strB_d));
    cudaErrchk(cudaFree(I_i));
    cudaErrchk(cudaFree(I_j));

    cudaErrchk(cudaFree(alAbeg_d));
    cudaErrchk(cudaFree(alBbeg_d));
    cudaErrchk(cudaFree(alAend_d));
    cudaErrchk(cudaFree(alBend_d));
  }
    auto                     end  = NOW;
    chrono::duration<double> diff = end - start;
    cout << "time = " << diff.count() << endl;

    string rstLine;
    ifstream rst_file(argv[3]);
    int k = 0, errors=0;
    if(rst_file.is_open())
    {
    while(getline(rst_file,rstLine))
    {
    string in = rstLine.substr(rstLine.find(":")+1, rstLine.size()-1);
    vector<int> valsVec;

    stringstream myStream(in);

    int val;
    while(myStream >> val){
      valsVec.push_back(val);
      if(myStream.peek() == ',')
        myStream.ignore();
    }

   int ref_st = valsVec[0];
   int ref_end = valsVec[1];
   int que_st = valsVec[2];
   int que_end = valsVec[3];

   if(test_Abeg[k] != ref_st || test_Aend[k] != ref_end || test_Bbeg[k] != que_st || test_Bend[k] != que_end){
  //  cout << "k:"<<k<<" startA=" << alAbeg[k] << ", endA=" << alAend[k]<<" startB=" << alBbeg[k] << ", endB=" << alBend[k]<<endl;
  //      cout << "corr:"<<k<<" corr_strtA=" << ref_st << ", corr_endA=" << ref_end<<" corr_startB=" << que_st << ", corr_endB=" << que_end<<endl;
     errors++;
   }
  //  cout <<ref_st<<" "<<ref_end<<" "<<ref_st<<" "<<ref_end<<endl;
k++;
    }
    cout <<"total errors:"<<errors<<endl;
    }

}// for iterations end here
//}// paralle pragma ends
//verifying correctness

// int error = 0;
//     for(int i = 0; i < NBLOCKS; i++){
//       if(alAbeg[i] != 189 || alAend[i] != 314 || alBbeg[i] != 1 || alBend[i] != 126){
//        cout << "i:"<<i<<" startA=" << alAbeg[i] << ", endA=" << alAend[i]<<" startB=" << alBbeg[i] << ", endB=" << alBend[i]<<endl;
//         error++;
//       }
//     }
//     cout <<"total errors:"<<error<<endl;

       //cout << " startA=" << alAbeg[0] << ", endA=" << alAend[0]<<" startB=" << alBbeg[0] << ", endB=" << alBend[0]<<endl;
    return 0;
}
