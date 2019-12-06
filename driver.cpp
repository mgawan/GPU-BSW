#include "driver.hpp"

#include <fstream>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>

void
callAlignKernel(std::vector<std::string> reads, std::vector<std::string> contigs,
                unsigned maxReadSize, unsigned maxContigSize, unsigned totalAlignments,
                short** gg_alAbeg, short** gg_alBbeg, short** gg_alAend,
                short** gg_alBend, char* rstFile)
{
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

    unsigned NBLOCKS = totalAlignments;

    unsigned alignmentsPerDevice = NBLOCKS / deviceCount;
    unsigned leftOver_device     = NBLOCKS % deviceCount;
    unsigned maxAligns           = alignmentsPerDevice + leftOver_device;

    long long totMemEst = maxContigSize * (long) maxAligns +
                          maxReadSize * (long) maxAligns /*+
                          maxMatrixSize * (long) maxAligns * sizeof(short) * 2*/
                          + (long) maxAligns * sizeof(short) * 4;

    long long estMem = totMemEst;
    int       its    = ceil(estMem / (prop[0].totalGlobalMem * 0.95));

    short* g_alAbeg = new short[NBLOCKS];
    short* g_alBbeg = new short[NBLOCKS];
    short* g_alAend = new short[NBLOCKS];
    short* g_alBend = new short[NBLOCKS];  // memory on CPU for copying the results

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

        short *alAbeg_d, *alBbeg_d, *alAend_d, *alBend_d;

        short* alAbeg = g_alAbeg + my_cpu_id * alignmentsPerDevice;
        short* alBbeg = g_alBbeg + my_cpu_id * alignmentsPerDevice;
        short* alAend = g_alAend + my_cpu_id * alignmentsPerDevice;
        short* alBend =
            g_alBend +
            my_cpu_id * alignmentsPerDevice;  // memory on CPU for copying the results

        thrust::host_vector<int>        offsetA(stringsPerIt + leftOvers);
        thrust::host_vector<int>        offsetB(stringsPerIt + leftOvers);
        thrust::device_vector<unsigned> vec_offsetA_d(stringsPerIt + leftOvers);
        thrust::device_vector<unsigned> vec_offsetB_d(stringsPerIt + leftOvers);

        unsigned* offsetA_d = thrust::raw_pointer_cast(&vec_offsetA_d[0]);
        unsigned* offsetB_d = thrust::raw_pointer_cast(&vec_offsetB_d[0]);

        // // copy back
        cudaErrchk(cudaMalloc(&alAbeg_d, (stringsPerIt + leftOvers) * sizeof(short)));
        cudaErrchk(cudaMalloc(&alBbeg_d, (stringsPerIt + leftOvers) * sizeof(short)));
        cudaErrchk(cudaMalloc(&alAend_d, (stringsPerIt + leftOvers) * sizeof(short)));
        cudaErrchk(cudaMalloc(&alBend_d, (stringsPerIt + leftOvers) * sizeof(short)));
        // //std::cout << "Iterations per GPU:"<<its<<std::endl;
        auto start2 = NOW;
        std::cout << "total its:" << its << std::endl;
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

            unsigned maxSize =
                (maxReadSize > maxContigSize) ? maxReadSize : maxContigSize;
            unsigned minSize =
                (maxReadSize < maxContigSize) ? maxReadSize : maxContigSize;

            unsigned totShmem = 3 * 3 * (minSize + 1) *
                                sizeof(short);  // +
                                                // 3 * minSize + (minSize & 1) + maxSize;

            unsigned alignmentPad = 4 + (4 - totShmem % 4);
            size_t   ShmemBytes =
                totShmem +
                alignmentPad; /*+ sizeof(int) * (maxContigSize + maxReadSize + 2*/

            if(ShmemBytes > 48000)
                cudaFuncSetAttribute(align_sequences_gpu,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     ShmemBytes);

            align_sequences_gpu<<<blocksLaunched, minSize, ShmemBytes>>>(
                strA_d, strB_d, offsetA_d, offsetB_d, alAbeg_d, alAend_d, alBbeg_d,
                alBend_d);
            //  std::cout <<"threads:"<<minSize<<std::endl;

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

            alAbeg += stringsPerIt;
            alBbeg += stringsPerIt;
            alAend += stringsPerIt;
            alBend += stringsPerIt;
            cudaErrchk(cudaFree(strA_d));
            cudaErrchk(cudaFree(strB_d));

        }  // for iterations end here
        auto                          end1  = NOW;
        std::chrono::duration<double> diff2 = end1 - start2;

        cudaErrchk(cudaFree(alAbeg_d));
        cudaErrchk(cudaFree(alBbeg_d));
        cudaErrchk(cudaFree(alAend_d));
        cudaErrchk(cudaFree(alBend_d));

#pragma omp barrier
    }  // paralle pragma ends
    auto                          end  = NOW;
    std::chrono::duration<double> diff = end - start;

    std::cout << "Total time:" << diff.count() << std::endl;

    *gg_alAbeg = g_alAbeg;
    *gg_alBbeg = g_alBbeg;
    *gg_alAend = g_alAend;
    *gg_alBend = g_alBend;
}

void
verificationTest(char* rstFile, short* g_alAbeg, short* g_alBbeg, short* g_alAend,
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
                std::cout << "actualAbeg:" << g_alAbeg[k] << " expected:" << ref_st
                          << std::endl;
                std::cout << "actualAend:" << g_alAend[k] << " expected:" << ref_end
                          << std::endl;
                std::cout << "actualBbeg:" << g_alBbeg[k] << " expected:" << que_st
                          << std::endl;

                std::cout << "actualBend:" << g_alBend[k] << " expected:" << que_end
                          << std::endl;
                std::cout << "index:" << k << std::endl;
            }
            k++;
        }
        if(errors == 0)
            std::cout << "VERIFICATION TEST PASSED" << std::endl;
        else
            std::cout << "ERRORS OCCURRED DURING VERIFICATION TEST" << std::endl;
    }
}
