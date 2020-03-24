#include "driver.hpp"
#include <fstream>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>

unsigned getMaxLength (std::vector<std::string> v)
{
  unsigned maxLength = 0;
  for(auto str : v){
    if(maxLength < str.length()){
      maxLength = str.length();
    }
  }

  return maxLength;
}

void
gpu_bsw_driver::kernel_driver_dna(std::vector<std::string> reads, std::vector<std::string> contigs, gpu_bsw_driver::alignment_results *alignments, short scores[4])
{
    short matchScore = scores[0], misMatchScore = scores[1], startGap = scores[2], extendGap = scores[3];
    unsigned maxContigSize = getMaxLength(contigs);
    unsigned maxReadSize = getMaxLength(reads);
    unsigned totalAlignments = contigs.size(); // assuming that read and contig vectors are same length

    std::cout<<"Alignments:"<<totalAlignments<<std::endl;
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cudaDeviceProp prop[deviceCount];
    for(int i = 0; i < deviceCount; i++)
      cudaGetDeviceProperties(&prop[i], 0);

    for(int i = 0; i < deviceCount; i++)
    {
        std::cout << "total Global Memory available on Device: " << i
             << " is:" << prop[i].totalGlobalMem << std::endl;
    }

    unsigned NBLOCKS             = totalAlignments;
  //  unsigned maxMatrixSize       = (maxContigSize + 1) * (maxReadSize + 1);
    unsigned alignmentsPerDevice = NBLOCKS / deviceCount;
    unsigned leftOver_device     = NBLOCKS % deviceCount;
    unsigned maxAligns           = alignmentsPerDevice + leftOver_device;

    long long totMemEst = maxContigSize * (long) maxAligns +
                          maxReadSize * (long) maxAligns /*+
                          maxMatrixSize * (long) maxAligns * sizeof(short) * 2*/ +
                          (long) maxAligns * sizeof(short) * (4+1); // + 1 for the top scores

    long long estMem = totMemEst;
    int       its    = ceil(estMem / (prop[0].totalGlobalMem * 0.90));
    // its = 3;

    short* g_alAbeg = new short[NBLOCKS];
    short* g_alBbeg = new short[NBLOCKS];
    short* g_alAend = new short[NBLOCKS];
    short* g_alBend = new short[NBLOCKS];  // memory on CPU for copying the results
    short* g_top_scores = new short[NBLOCKS];

    auto start = NOW;
#pragma omp parallel
    {
        int my_cpu_id = omp_get_thread_num();
        cudaSetDevice(my_cpu_id);
        int myGPUid;
        cudaGetDevice(&myGPUid);
        int BLOCKS_l = alignmentsPerDevice;
        if(my_cpu_id == deviceCount - 1)
            BLOCKS_l += leftOver_device;

        unsigned leftOvers    = BLOCKS_l % its;
        unsigned stringsPerIt = BLOCKS_l / its;

        short *alAbeg_d, *alBbeg_d, *alAend_d, *alBend_d, *top_scores_d;

        short* alAbeg = g_alAbeg + my_cpu_id * alignmentsPerDevice;
        short* alBbeg = g_alBbeg + my_cpu_id * alignmentsPerDevice;
        short* alAend = g_alAend + my_cpu_id * alignmentsPerDevice;
        short* alBend =
            g_alBend +
            my_cpu_id * alignmentsPerDevice;  // memory on CPU for copying the results

        short* top_scores_cpu = g_top_scores + my_cpu_id * alignmentsPerDevice;


        thrust::host_vector<int>        offsetA(stringsPerIt + leftOvers);
        thrust::host_vector<int>        offsetB(stringsPerIt + leftOvers);
        thrust::device_vector<unsigned> vec_offsetA_d(stringsPerIt + leftOvers);
        thrust::device_vector<unsigned> vec_offsetB_d(stringsPerIt + leftOvers);

        unsigned* offsetA_d = thrust::raw_pointer_cast(&vec_offsetA_d[0]);
        unsigned* offsetB_d = thrust::raw_pointer_cast(&vec_offsetB_d[0]);

        // cudaErrchk(
        //     cudaMalloc(&I_i, maxMatrixSize * (stringsPerIt + leftOvers) * sizeof(short)));
        // cudaErrchk(
        //     cudaMalloc(&I_j, maxMatrixSize * (stringsPerIt + leftOvers) * sizeof(short)));

        // // copy back
        cudaErrchk(cudaMalloc(&alAbeg_d, (stringsPerIt + leftOvers) * sizeof(short)));
        cudaErrchk(cudaMalloc(&alBbeg_d, (stringsPerIt + leftOvers) * sizeof(short)));
        cudaErrchk(cudaMalloc(&alAend_d, (stringsPerIt + leftOvers) * sizeof(short)));
        cudaErrchk(cudaMalloc(&alBend_d, (stringsPerIt + leftOvers) * sizeof(short)));
        cudaErrchk(cudaMalloc(&top_scores_d, (stringsPerIt + leftOvers) * sizeof(short)));

        auto start2 = NOW;
        std::cout << "total iterations:" << its << std::endl;
        for(int perGPUIts = 0; perGPUIts < its; perGPUIts++)
        {
            int                                      blocksLaunched = 0;
            std::vector<std::string>::const_iterator beginAVec;
            std::vector<std::string>::const_iterator endAVec;
            std::vector<std::string>::const_iterator beginBVec;
            std::vector<std::string>::const_iterator endBVec;
            if(perGPUIts == its - 1)
            {
                beginAVec = contigs.begin() + ((alignmentsPerDevice * my_cpu_id) +
                                               perGPUIts * stringsPerIt);
                endAVec =
                    contigs.begin() +
                    ((alignmentsPerDevice * my_cpu_id) + (perGPUIts + 1) * stringsPerIt) +
                    leftOvers;  // so that each openmp thread has a copy of strings it
                                // needs to align
                beginBVec = reads.begin() + ((alignmentsPerDevice * my_cpu_id) +
                                             perGPUIts * stringsPerIt);
                endBVec =
                    reads.begin() +
                    ((alignmentsPerDevice * my_cpu_id) + (perGPUIts + 1) * stringsPerIt) +
                    leftOvers;  // so that each openmp thread has a copy of strings it
                                // needs to align

                blocksLaunched = stringsPerIt + leftOvers;
            }
            else
            {
                beginAVec = contigs.begin() + ((alignmentsPerDevice * my_cpu_id) +
                                               perGPUIts * stringsPerIt);
                endAVec =
                    contigs.begin() + (alignmentsPerDevice * my_cpu_id) +
                    (perGPUIts + 1) * stringsPerIt;  // so that each openmp thread has a
                                                     // copy of strings it needs to align
                beginBVec = reads.begin() + ((alignmentsPerDevice * my_cpu_id) +
                                             perGPUIts * stringsPerIt);
                endBVec =
                    reads.begin() + (alignmentsPerDevice * my_cpu_id) +
                    (perGPUIts + 1) * stringsPerIt;  // so that each openmp thread has a
                                                     // copy of strings it needs to align
                blocksLaunched = stringsPerIt;
            }

            std::vector<std::string> sequencesA(beginAVec, endAVec);
            std::vector<std::string> sequencesB(beginBVec, endBVec);

            for(int i = 0; i < sequencesA.size(); i++)
            {
                offsetA[i] = sequencesA[i].size();
            }

            for(int i = 0; i < sequencesB.size(); i++)
            {
                offsetB[i] = sequencesB[i].size();
            }

            vec_offsetA_d = offsetA;
            vec_offsetB_d = offsetB;

            thrust::inclusive_scan(vec_offsetA_d.begin(), vec_offsetA_d.end(),
                                   vec_offsetA_d.begin());
            thrust::inclusive_scan(vec_offsetB_d.begin(), vec_offsetB_d.end(),
                                   vec_offsetB_d.begin());

            unsigned totalLengthA = vec_offsetA_d[sequencesA.size() - 1];
            unsigned totalLengthB = vec_offsetB_d[sequencesB.size() - 1];

            unsigned offsetSumA = 0;
            unsigned offsetSumB = 0;

            char* strA = new char[totalLengthA];
            char* strB = new char[totalLengthB];
            for(int i = 0; i < sequencesA.size(); i++)
            {
                char* seqptrA =
                    strA + offsetSumA;  // vec_offsetA_d[i] - sequencesA[i].size();
                memcpy(seqptrA, sequencesA[i].c_str(), sequencesA[i].size());

                char* seqptrB =
                    strB + offsetSumB;  // vec_offsetB_d[i] - sequencesB[i].size();
                memcpy(seqptrB, sequencesB[i].c_str(), sequencesB[i].size());
                offsetSumA += sequencesA[i].size();
                offsetSumB += sequencesB[i].size();
            }

            char *strA_d, *strB_d;
            cudaErrchk(cudaMalloc(&strA_d, totalLengthA * sizeof(char)));
            cudaErrchk(cudaMalloc(&strB_d, totalLengthB * sizeof(char)));

            cudaErrchk(cudaMemcpy(strA_d, strA, totalLengthA * sizeof(char),
                                  cudaMemcpyHostToDevice));
            cudaErrchk(cudaMemcpy(strB_d, strB, totalLengthB * sizeof(char),
                                  cudaMemcpyHostToDevice));

          // unsigned maxSize = (maxReadSize > maxContigSize) ? maxReadSize : maxContigSize;
           unsigned minSize = (maxReadSize < maxContigSize) ? maxReadSize : maxContigSize;

            unsigned totShmem = 3 * (minSize + 1) * sizeof(short);// +
                                //3 * minSize + (minSize & 1) + maxSize;

            unsigned alignmentPad = 4 + (4 - totShmem % 4);
            size_t   ShmemBytes = totShmem + alignmentPad; /*+ sizeof(int) * (maxContigSize + maxReadSize + 2*/

            if(ShmemBytes > 48000)
                cudaFuncSetAttribute(gpu_bsw::sequence_dna_kernel,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     ShmemBytes);

            gpu_bsw::sequence_dna_kernel<<<blocksLaunched, minSize, ShmemBytes>>>(
                strA_d, strB_d, offsetA_d, offsetB_d, alAbeg_d,
                alAend_d, alBbeg_d, alBend_d, top_scores_d, matchScore, misMatchScore, startGap, extendGap);

            cudaErrchk(cudaMemcpy(alAbeg, alAbeg_d, blocksLaunched * sizeof(short),
                                  cudaMemcpyDeviceToHost));
            cudaErrchk(cudaMemcpy(alBbeg, alBbeg_d, blocksLaunched * sizeof(short),
                                  cudaMemcpyDeviceToHost));
            cudaErrchk(cudaMemcpy(alAend, alAend_d, blocksLaunched * sizeof(short),
                                  cudaMemcpyDeviceToHost));
            cudaErrchk(
                cudaMemcpy(alBend, alBend_d, blocksLaunched * sizeof(short),
                           cudaMemcpyDeviceToHost));  // this does not cause the error
                                                      // the other three lines do.
            cudaErrchk(cudaMemcpy(top_scores_cpu, top_scores_d, blocksLaunched * sizeof(short),
                                  cudaMemcpyDeviceToHost));

            //}
            alAbeg += stringsPerIt;  // perGPUIts;//*stringsPerIt;
            alBbeg += stringsPerIt;  //;//*stringsPerIt;
            alAend += stringsPerIt;  //;//*stringsPerIt;
            alBend += stringsPerIt;  //;//*stringsPerIt;

            top_scores_cpu += stringsPerIt;

            cudaErrchk(cudaFree(strA_d));
            cudaErrchk(cudaFree(strB_d));
            //}
        }  // for iterations end here
        auto                          end1  = NOW;
        std::chrono::duration<double> diff2 = end1 - start2;

        // cudaErrchk(cudaFree(I_i));
        // cudaErrchk(cudaFree(I_j));

        cudaErrchk(cudaFree(alAbeg_d));
        cudaErrchk(cudaFree(alBbeg_d));
        cudaErrchk(cudaFree(alAend_d));
        cudaErrchk(cudaFree(alBend_d));
        cudaErrchk(cudaFree(top_scores_d));


#pragma omp barrier
    }  // paralle pragma ends
    auto                          end  = NOW;
    std::chrono::duration<double> diff = end - start;

    std::cout << "Total time:" << diff.count() << std::endl;

    alignments->g_alAbeg = g_alAbeg;
    alignments->g_alBbeg = g_alBbeg;
    alignments->g_alAend = g_alAend;
    alignments->g_alBend = g_alBend;
    alignments->top_scores = g_top_scores;
}


// void
// gpu_bsw_driver::kernel_driver_aa(std::vector<std::string> reads, std::vector<std::string> contigs,
//                 short** gg_alAbeg, short** gg_alBbeg, short** gg_alAend,
//                 short** gg_alBend, short** gg_top_scores, short scoring_matrix[], short openGap, short extendGap)
void
gpu_bsw_driver::kernel_driver_aa(std::vector<std::string> reads, std::vector<std::string> contigs, gpu_bsw_driver::alignment_results *alignments, short scoring_matrix[], short openGap, short extendGap)
{


  unsigned maxContigSize = getMaxLength(contigs);
  unsigned maxReadSize = getMaxLength(reads);
  unsigned totalAlignments = contigs.size(); // assuming that read and contig vectors are same length

  std::cout << "max Contig Length:"<<maxContigSize<<std::endl;
  std::cout << "max read length:"<<maxReadSize<<std::endl;
    std::cout<<"Alignments:"<<totalAlignments<<std::endl;
  short encoding_matrix[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                          0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                          0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                          23,0,0,0,0,0,0,0,0,0,0,0,0,0,
                          0,0,0,0,0,0,0,0,0,0,20,4,3,6,
                          13,7,8,9,0,11,10,12,2,0,14,5,
                          1,15,16,0,19,17,22,18,21};

  short* d_encoding_matrix;
  short* d_scoring_matrix;

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  cudaDeviceProp prop[deviceCount];
  for(int i = 0; i < deviceCount; i++)
    cudaGetDeviceProperties(&prop[i], 0);

  for(int i = 0; i < deviceCount; i++)
  {
      std::cout << "total Global Memory available on Device: " << i
           << " is:" << prop[i].totalGlobalMem << std::endl;
  }

  unsigned NBLOCKS             = totalAlignments;
//  unsigned maxMatrixSize       = (maxContigSize + 1) * (maxReadSize + 1);
  unsigned alignmentsPerDevice = NBLOCKS / deviceCount;
  unsigned leftOver_device     = NBLOCKS % deviceCount;
  unsigned maxAligns           = alignmentsPerDevice + leftOver_device;

  long long totMemEst = maxContigSize * (long) maxAligns +
                        maxReadSize * (long) maxAligns /*+
                        maxMatrixSize * (long) maxAligns * sizeof(short) * 2*/ +
                        (long) maxAligns * sizeof(short) * (4+1);// +1 for scores

  long long estMem = totMemEst;
  int       its    = ceil(estMem / (prop[0].totalGlobalMem * 0.90));
 // its = 5;

  short* g_alAbeg = new short[NBLOCKS];
  short* g_alBbeg = new short[NBLOCKS];
  short* g_alAend = new short[NBLOCKS];
  short* g_alBend = new short[NBLOCKS];  // memory on CPU for copying the results
  short* g_top_scores = new short[NBLOCKS];

  auto start = NOW;
#pragma omp parallel
  {
      int my_cpu_id = omp_get_thread_num();
      cudaSetDevice(my_cpu_id);
      int myGPUid;
      cudaGetDevice(&myGPUid);
      int BLOCKS_l = alignmentsPerDevice;
      if(my_cpu_id == deviceCount - 1)
          BLOCKS_l += leftOver_device;

      unsigned leftOvers    = BLOCKS_l % its;
      unsigned stringsPerIt = BLOCKS_l / its;

      short *alAbeg_d, *alBbeg_d, *alAend_d, *alBend_d, *top_scores_d;

      short* alAbeg = g_alAbeg + my_cpu_id * alignmentsPerDevice;
      short* alBbeg = g_alBbeg + my_cpu_id * alignmentsPerDevice;
      short* alAend = g_alAend + my_cpu_id * alignmentsPerDevice;
      short* alBend = g_alBend + my_cpu_id * alignmentsPerDevice;  // memory on CPU for copying the results
      short *top_scores_cpu = g_top_scores + my_cpu_id * alignmentsPerDevice;

      thrust::host_vector<int>        offsetA(stringsPerIt + leftOvers);
      thrust::host_vector<int>        offsetB(stringsPerIt + leftOvers);
      thrust::device_vector<unsigned> vec_offsetA_d(stringsPerIt + leftOvers);
      thrust::device_vector<unsigned> vec_offsetB_d(stringsPerIt + leftOvers);

      unsigned* offsetA_d = thrust::raw_pointer_cast(&vec_offsetA_d[0]);
      unsigned* offsetB_d = thrust::raw_pointer_cast(&vec_offsetB_d[0]);

      // cudaErrchk(
      //     cudaMalloc(&I_i, maxMatrixSize * (stringsPerIt + leftOvers) * sizeof(short)));
      // cudaErrchk(
      //     cudaMalloc(&I_j, maxMatrixSize * (stringsPerIt + leftOvers) * sizeof(short)));

      // // copy back
      cudaErrchk(cudaMalloc(&alAbeg_d, (stringsPerIt + leftOvers) * sizeof(short)));
      cudaErrchk(cudaMalloc(&alBbeg_d, (stringsPerIt + leftOvers) * sizeof(short)));
      cudaErrchk(cudaMalloc(&alAend_d, (stringsPerIt + leftOvers) * sizeof(short)));
      cudaErrchk(cudaMalloc(&alBend_d, (stringsPerIt + leftOvers) * sizeof(short)));
      cudaErrchk(cudaMalloc(&top_scores_d, (stringsPerIt + leftOvers) * sizeof(short)));

      cudaErrchk(cudaMalloc(&d_encoding_matrix, ENCOD_MAT_SIZE * sizeof(short)));
      cudaErrchk(cudaMalloc(&d_scoring_matrix, SCORE_MAT_SIZE * sizeof(short)));

      auto start2 = NOW;
      std::cout << "total iterations:" << its << std::endl;
      for(int perGPUIts = 0; perGPUIts < its; perGPUIts++)
      {
          int                                      blocksLaunched = 0;
          std::vector<std::string>::const_iterator beginAVec;
          std::vector<std::string>::const_iterator endAVec;
          std::vector<std::string>::const_iterator beginBVec;
          std::vector<std::string>::const_iterator endBVec;
          if(perGPUIts == its - 1)
          {
              beginAVec = contigs.begin() + ((alignmentsPerDevice * my_cpu_id) +
                                             perGPUIts * stringsPerIt);
              endAVec =
                  contigs.begin() +
                  ((alignmentsPerDevice * my_cpu_id) + (perGPUIts + 1) * stringsPerIt) +
                  leftOvers;  // so that each openmp thread has a copy of strings it
                              // needs to align
              beginBVec = reads.begin() + ((alignmentsPerDevice * my_cpu_id) +
                                           perGPUIts * stringsPerIt);
              endBVec =
                  reads.begin() +
                  ((alignmentsPerDevice * my_cpu_id) + (perGPUIts + 1) * stringsPerIt) +
                  leftOvers;  // so that each openmp thread has a copy of strings it
                              // needs to align

              blocksLaunched = stringsPerIt + leftOvers;
          }
          else
          {
              beginAVec = contigs.begin() + ((alignmentsPerDevice * my_cpu_id) +
                                             perGPUIts * stringsPerIt);
              endAVec =
                  contigs.begin() + (alignmentsPerDevice * my_cpu_id) +
                  (perGPUIts + 1) * stringsPerIt;  // so that each openmp thread has a
                                                   // copy of strings it needs to align
              beginBVec = reads.begin() + ((alignmentsPerDevice * my_cpu_id) +
                                           perGPUIts * stringsPerIt);
              endBVec =
                  reads.begin() + (alignmentsPerDevice * my_cpu_id) +
                  (perGPUIts + 1) * stringsPerIt;  // so that each openmp thread has a
                                                   // copy of strings it needs to align
              blocksLaunched = stringsPerIt;
          }

          std::vector<std::string> sequencesA(beginAVec, endAVec);
          std::vector<std::string> sequencesB(beginBVec, endBVec);

          for(int i = 0; i < sequencesA.size(); i++)
          {
              offsetA[i] = sequencesA[i].size();
          }

          for(int i = 0; i < sequencesB.size(); i++)
          {
              offsetB[i] = sequencesB[i].size();
          }

          vec_offsetA_d = offsetA;
          vec_offsetB_d = offsetB;

          thrust::inclusive_scan(vec_offsetA_d.begin(), vec_offsetA_d.end(),
                                 vec_offsetA_d.begin());
          thrust::inclusive_scan(vec_offsetB_d.begin(), vec_offsetB_d.end(),
                                 vec_offsetB_d.begin());

          unsigned totalLengthA = vec_offsetA_d[sequencesA.size() - 1];
          unsigned totalLengthB = vec_offsetB_d[sequencesB.size() - 1];

          unsigned offsetSumA = 0;
          unsigned offsetSumB = 0;

          char* strA = new char[totalLengthA];
          char* strB = new char[totalLengthB];
          for(int i = 0; i < sequencesA.size(); i++)
          {
              char* seqptrA =
                  strA + offsetSumA;  // vec_offsetA_d[i] - sequencesA[i].size();
              memcpy(seqptrA, sequencesA[i].c_str(), sequencesA[i].size());

              char* seqptrB =
                  strB + offsetSumB;  // vec_offsetB_d[i] - sequencesB[i].size();
              memcpy(seqptrB, sequencesB[i].c_str(), sequencesB[i].size());
              offsetSumA += sequencesA[i].size();
              offsetSumB += sequencesB[i].size();
          }

          char *strA_d, *strB_d;
          cudaErrchk(cudaMalloc(&strA_d, totalLengthA * sizeof(char)));
          cudaErrchk(cudaMalloc(&strB_d, totalLengthB * sizeof(char)));

          cudaErrchk(cudaMemcpy(strA_d, strA, totalLengthA * sizeof(char),
                                cudaMemcpyHostToDevice));
          cudaErrchk(cudaMemcpy(strB_d, strB, totalLengthB * sizeof(char),
                                cudaMemcpyHostToDevice));

          cudaErrchk(cudaMemcpy(d_encoding_matrix, encoding_matrix, 91 * sizeof(short),
                                cudaMemcpyHostToDevice));

          cudaErrchk(cudaMemcpy(d_scoring_matrix, scoring_matrix, 576 * sizeof(short),
                                cudaMemcpyHostToDevice));

        // unsigned maxSize = (maxReadSize > maxContigSize) ? maxReadSize : maxContigSize;
         unsigned minSize = (maxReadSize < maxContigSize) ? maxReadSize : maxContigSize;

          unsigned totShmem = 3 * (minSize + 1) * sizeof(short);// +
                              //3 * minSize + (minSize & 1) + maxSize;

          unsigned alignmentPad = 4 + (4 - totShmem % 4);
          size_t   ShmemBytes = totShmem + alignmentPad; /*+ sizeof(int) * (maxContigSize + maxReadSize + 2*/

          if(ShmemBytes > 48000)
              cudaFuncSetAttribute(gpu_bsw::sequence_aa_kernel,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   ShmemBytes);

          gpu_bsw::sequence_aa_kernel<<<blocksLaunched, minSize, ShmemBytes>>>(
              strA_d, strB_d, offsetA_d, offsetB_d, alAbeg_d,
              alAend_d, alBbeg_d, alBend_d, top_scores_d,openGap, extendGap, d_scoring_matrix, d_encoding_matrix);
              std::cout <<"threads:"<<minSize<<std::endl;

          cudaErrchk(cudaMemcpy(alAbeg, alAbeg_d, blocksLaunched * sizeof(short),
                                cudaMemcpyDeviceToHost));
          cudaErrchk(cudaMemcpy(alBbeg, alBbeg_d, blocksLaunched * sizeof(short),
                                cudaMemcpyDeviceToHost));
          cudaErrchk(cudaMemcpy(alAend, alAend_d, blocksLaunched * sizeof(short),
                                cudaMemcpyDeviceToHost));
          cudaErrchk(
              cudaMemcpy(alBend, alBend_d, blocksLaunched * sizeof(short),
                         cudaMemcpyDeviceToHost));
          cudaErrchk(cudaMemcpy(top_scores_cpu, top_scores_d, blocksLaunched * sizeof(short), cudaMemcpyDeviceToHost));


          //}
          alAbeg += stringsPerIt;  // perGPUIts;//*stringsPerIt;
          alBbeg += stringsPerIt;  //;//*stringsPerIt;
          alAend += stringsPerIt;  //;//*stringsPerIt;
          alBend += stringsPerIt;  //;//*stringsPerIt;
          top_scores_cpu += stringsPerIt;

          cudaErrchk(cudaFree(strA_d));
          cudaErrchk(cudaFree(strB_d));
          //}
      }  // for iterations end here
      auto                          end1  = NOW;
      std::chrono::duration<double> diff2 = end1 - start2;

      // cudaErrchk(cudaFree(I_i));
      // cudaErrchk(cudaFree(I_j));

      cudaErrchk(cudaFree(alAbeg_d));
      cudaErrchk(cudaFree(alBbeg_d));
      cudaErrchk(cudaFree(alAend_d));
      cudaErrchk(cudaFree(alBend_d));
      // cudaErrchk(cudaFree(d_encoding_matrix));
      // cudaErrchk(cudaFree(d_scoring_matrix));

      cudaErrchk(cudaFree(top_scores_d));


#pragma omp barrier
  }  // paralle pragma ends
  auto                          end  = NOW;
  std::chrono::duration<double> diff = end - start;

  std::cout << "Total time:" << diff.count() << std::endl;


  alignments->g_alAbeg = g_alAbeg;
  alignments->g_alBbeg = g_alBbeg;
  alignments->g_alAend = g_alAend;
  alignments->g_alBend = g_alBend;
  alignments->top_scores = g_top_scores;
}


void
gpu_bsw_driver::verificationTest(std::string rstFile, short* g_alAbeg, short* g_alBbeg, short* g_alAend,
                 short* g_alBend)
{
    std::string   rstLine;
    std::ifstream rst_file(rstFile);
    int           k = 0, errors = 0;
    if(rst_file.is_open())
    {
        while(getline(rst_file, rstLine))
        {
            std::string in = rstLine.substr(rstLine.find(":") + 1, rstLine.size() - 1);
            std::vector<int> valsVec;

            std::stringstream myStream(in);

            int val;
            while(myStream >> val)
            {
                valsVec.push_back(val);
                if(myStream.peek() == ',')
                    myStream.ignore();
            }

            int ref_st  = valsVec[0];
            int ref_end = valsVec[1];
            int que_st  = valsVec[2];
            int que_end = valsVec[3];

            if(g_alAbeg[k] != ref_st || g_alAend[k] != ref_end || g_alBbeg[k] != que_st ||
               g_alBend[k] != que_end)
            {
                errors++;
                 std::cout<<"actualAbeg:"<<g_alAbeg[k]<<" expected:"<<ref_st<<std::endl;
                std::cout<<"actualAend:"<<g_alAend[k]<<" expected:"<<ref_end<<std::endl;
                 std::cout<<"actualBbeg:"<<g_alBbeg[k]<<" expected:"<<que_st<<std::endl;

                std::cout<<"actualBend:"<<g_alBend[k]<<" expected:"<<que_end<<std::endl;
std::cout<<"index:"<<k<<std::endl;
            }
            k++;
        }
        if(errors == 0)
            std::cout << "VERIFICATION TEST PASSED" << std::endl;
        else
            std::cout << "ERRORS OCCURRED DURING VERIFICATION TEST" << std::endl;
            std::cout << "ERRORS:"<<errors<<std::endl;
    }
}
