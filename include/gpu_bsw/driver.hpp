#ifndef DRIVER_HPP
#define DRIVER_HPP

#include <gpu_bsw/alignments.hpp>
#include <gpu_bsw/driver.hpp>
#include <gpu_bsw/kernel.hpp>
#include <gpu_bsw/page_locked_string.hpp>
#include <gpu_bsw/timer.hpp>
#include <gpu_bsw/utils.hpp>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>

constexpr int NSTREAMS = 2;


namespace gpu_bsw_driver {
  // for storing the alignment results
  struct alignment_results{
    short* ref_begin = nullptr;
    short* query_begin = nullptr;
    short* ref_end = nullptr;
    short* query_end = nullptr;
    short* top_scores = nullptr;
  };
}

void initialize_alignments(gpu_bsw_driver::alignment_results *alignments, int max_alignments);
void free_alignments(gpu_bsw_driver::alignment_results *alignments);
void asynch_mem_copies_htd(gpu_alignments* gpu_data, unsigned* offsetA_h, unsigned* offsetB_h, char* strA, char* strA_d, char* strB, char* strB_d, unsigned half_length_A, unsigned half_length_B, unsigned totalLengthA, unsigned totalLengthB, int sequences_per_stream, int sequences_stream_leftover, const std::array<cudaStream_t,2> &streams_cuda);
void asynch_mem_copies_dth_mid(gpu_alignments* gpu_data, short* alAend, short* alBend, int sequences_per_stream, int sequences_stream_leftover, const std::array<cudaStream_t,2> &streams_cuda);
void asynch_mem_copies_dth(gpu_alignments* gpu_data, short* alAbeg, short* alBbeg, short* top_scores_cpu, int sequences_per_stream, int sequences_stream_leftover, const std::array<cudaStream_t,2> &streams_cuda);



namespace gpu_bsw_driver{

template<DataType DT>
void kernel_driver(
  const std::vector<std::string> &reads,
  const std::vector<std::string> &contigs,
  gpu_bsw_driver::alignment_results *const alignments,
  const short scoring_matrix[],
  const short openGap,
  const short extendGap
){
    const auto maxContigSize = getMaxLength(contigs);
    const auto maxReadSize = getMaxLength(reads);
    unsigned totalAlignments = contigs.size(); // assuming that read and contig vectors are same length

    //This matrix is used only by the RNA kernel
    constexpr short encoding_matrix[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                         0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                         0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                        23,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                         0,0,0,0,0,0,0,0,0,0,20,4,3,6,
                                        13,7,8,9,0,11,10,12,2,0,14,5,
                                         1,15,16,0,19,17,22,18,21};

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    omp_set_num_threads(deviceCount); //one OMP thread per GPU
    std::cout<<"Number of available GPUs:"<<deviceCount<<"\n";

    cudaDeviceProp prop[deviceCount];
    for(int i = 0; i < deviceCount; i++)
      cudaGetDeviceProperties(&prop[i], 0);

    unsigned NBLOCKS             = totalAlignments;
    unsigned alignmentsPerDevice = NBLOCKS / deviceCount;
    unsigned leftOver_device     = NBLOCKS % deviceCount;
    unsigned max_per_device = alignmentsPerDevice + leftOver_device;
    int       its    = (max_per_device>20000)?(ceil((float)max_per_device/20000)):1;
    initialize_alignments(alignments, totalAlignments); // pinned memory allocation

    Timer timer_total;
    timer_total.start();

    #pragma omp parallel
    {
        const int my_cpu_id = omp_get_thread_num();
        cudaSetDevice(my_cpu_id);
        int myGPUid;
        cudaGetDevice(&myGPUid);
        Timer timer_cpu;

        std::array<cudaStream_t, 2> streams_cuda;
        for(auto &stream: streams_cuda){
          cudaStreamCreate(&stream);
        }

        int BLOCKS_l = alignmentsPerDevice;
        if(my_cpu_id == deviceCount - 1)
            BLOCKS_l += leftOver_device;
        if(my_cpu_id == 0)std::cout<<"Number of GPUs being used:"<<omp_get_num_threads()<<"\n";
        unsigned leftOvers    = BLOCKS_l % its;
        unsigned stringsPerIt = BLOCKS_l / its;
        gpu_alignments gpu_data(stringsPerIt + leftOvers); // gpu mallocs

        short *d_scoring_matrix;
        //This matrix is only used by the RNA kernel
        short *d_encoding_matrix;

        if(DT==DataType::RNA){
          d_encoding_matrix = DeviceMalloc<short>(ENCOD_MAT_SIZE, encoding_matrix);
        }
        d_scoring_matrix  = DeviceMalloc<short>(SCORE_MAT_SIZE, scoring_matrix);

        short* alAbeg         = alignments->ref_begin   + my_cpu_id * alignmentsPerDevice;
        short* alBbeg         = alignments->query_begin + my_cpu_id * alignmentsPerDevice;
        short* alAend         = alignments->ref_end     + my_cpu_id * alignmentsPerDevice;
        short* alBend         = alignments->query_begin + my_cpu_id * alignmentsPerDevice;  // memory on CPU for copying the results
        short* top_scores_cpu = alignments->top_scores  + my_cpu_id * alignmentsPerDevice;

        unsigned *const offsetA_h = PageLockedMalloc<unsigned>(stringsPerIt + leftOvers);
        unsigned *const offsetB_h = PageLockedMalloc<unsigned>(stringsPerIt + leftOvers);

        char *const strA_d = DeviceMalloc<char>(maxContigSize * (stringsPerIt + leftOvers));
        char *const strB_d = DeviceMalloc<char>(maxReadSize   * (stringsPerIt + leftOvers));

        auto strA = PageLockedString(maxContigSize * (stringsPerIt + leftOvers));
        auto strB = PageLockedString(maxReadSize   * (stringsPerIt + leftOvers));

        Timer timer_packing;

        std::cout<<"loop begin\n";
        for(int perGPUIts = 0; perGPUIts < its; perGPUIts++)
        {
            timer_packing.start();
            size_t blocksLaunched = stringsPerIt;

            // so that each openmp thread has a copy of strings it needs to align
            auto beginAVec = contigs.cbegin() + alignmentsPerDevice * my_cpu_id + perGPUIts       * stringsPerIt;
            auto endAVec   = contigs.cbegin() + alignmentsPerDevice * my_cpu_id + (perGPUIts + 1) * stringsPerIt;
            auto beginBVec = reads.cbegin()   + alignmentsPerDevice * my_cpu_id + perGPUIts       * stringsPerIt;
            auto endBVec   = reads.cbegin()   + alignmentsPerDevice * my_cpu_id + (perGPUIts + 1) * stringsPerIt;
            if(perGPUIts == its - 1)
            {
              blocksLaunched += leftOvers;
              endAVec += leftOvers;
              endBVec += leftOvers;
            }

            std::vector<std::string> sequencesA(beginAVec, endAVec);
            std::vector<std::string> sequencesB(beginBVec, endBVec);
            unsigned running_sum = 0;
            const auto sequences_per_stream = (blocksLaunched) / NSTREAMS;
            const auto sequences_stream_leftover = (blocksLaunched) % NSTREAMS;
            unsigned half_length_A = 0;
            unsigned half_length_B = 0;

            timer_cpu.start();

            for(size_t i = 0; i < sequencesA.size(); i++)
            {
                running_sum +=sequencesA[i].size();
                offsetA_h[i] = running_sum;//sequencesA[i].size();
                if(i == sequences_per_stream - 1){
                    half_length_A = running_sum;
                    running_sum = 0;
                }
            }
            unsigned totalLengthA = half_length_A + offsetA_h[sequencesA.size() - 1];

            running_sum = 0;
            for(size_t i = 0; i < sequencesB.size(); i++)
            {
                running_sum +=sequencesB[i].size();
                offsetB_h[i] = running_sum; //sequencesB[i].size();
                if(i == sequences_per_stream - 1){
                  half_length_B = running_sum;
                  running_sum = 0;
                }
            }
            unsigned totalLengthB = half_length_B + offsetB_h[sequencesB.size() - 1];

            timer_cpu.stop();

            strA.clear();
            strB.clear();
            for(size_t i = 0; i < sequencesA.size(); i++)
            {
              strA += sequencesA.at(i);
              strB += sequencesB.at(i);
            }

            timer_packing.stop();

            asynch_mem_copies_htd(&gpu_data, offsetA_h, offsetB_h, strA.data(), strA_d, strB.data(), strB_d, half_length_A, half_length_B, totalLengthA, totalLengthB, sequences_per_stream, sequences_stream_leftover, streams_cuda);
            unsigned minSize = (maxReadSize < maxContigSize) ? maxReadSize : maxContigSize;
            unsigned totShmem = 3 * (minSize + 1) * sizeof(short);
            unsigned alignmentPad = 4 + (4 - totShmem % 4);
            size_t   ShmemBytes = totShmem + alignmentPad;
            if(ShmemBytes > 48000 && DT==DataType::DNA)
                cudaFuncSetAttribute(gpu_bsw::sequence_kernel<DT>, cudaFuncAttributeMaxDynamicSharedMemorySize, ShmemBytes);

            gpu_bsw::sequence_kernel<DT><<<sequences_per_stream, minSize, ShmemBytes, streams_cuda[0]>>>(
                strA_d, strB_d, gpu_data.offset_ref_gpu, gpu_data.offset_query_gpu, gpu_data.ref_start_gpu,
                gpu_data.ref_end_gpu, gpu_data.query_start_gpu, gpu_data.query_end_gpu, gpu_data.scores_gpu,
                openGap, extendGap, d_scoring_matrix, d_encoding_matrix);
            cudaErrchk(cudaGetLastError());

            gpu_bsw::sequence_kernel<DT><<<sequences_per_stream + sequences_stream_leftover, minSize, ShmemBytes, streams_cuda[1]>>>(
                strA_d + half_length_A, strB_d + half_length_B, gpu_data.offset_ref_gpu + sequences_per_stream, gpu_data.offset_query_gpu + sequences_per_stream,
                gpu_data.ref_start_gpu + sequences_per_stream, gpu_data.ref_end_gpu + sequences_per_stream, gpu_data.query_start_gpu + sequences_per_stream, gpu_data.query_end_gpu + sequences_per_stream,
                gpu_data.scores_gpu + sequences_per_stream, openGap, extendGap, d_scoring_matrix, d_encoding_matrix);
            cudaErrchk(cudaGetLastError());

            // copyin back end index so that we can find new min
            asynch_mem_copies_dth_mid(&gpu_data, alAend, alBend, sequences_per_stream, sequences_stream_leftover, streams_cuda);

            cudaStreamSynchronize (streams_cuda[0]);
            cudaStreamSynchronize (streams_cuda[1]);

            timer_cpu.start();
            int newMin = get_new_min_length(alAend, alBend, blocksLaunched); // find the new largest of smaller lengths
            timer_cpu.stop();

            gpu_bsw::sequence_reverse<DT><<<sequences_per_stream, newMin, ShmemBytes, streams_cuda[0]>>>(
                  strA_d, strB_d, gpu_data.offset_ref_gpu, gpu_data.offset_query_gpu, gpu_data.ref_start_gpu,
                  gpu_data.ref_end_gpu, gpu_data.query_start_gpu, gpu_data.query_end_gpu, gpu_data.scores_gpu, openGap, extendGap, d_scoring_matrix, d_encoding_matrix);
            cudaErrchk(cudaGetLastError());

            gpu_bsw::sequence_reverse<DT><<<sequences_per_stream + sequences_stream_leftover, newMin, ShmemBytes, streams_cuda[1]>>>(
                  strA_d + half_length_A, strB_d + half_length_B, gpu_data.offset_ref_gpu + sequences_per_stream, gpu_data.offset_query_gpu + sequences_per_stream ,
                  gpu_data.ref_start_gpu + sequences_per_stream, gpu_data.ref_end_gpu + sequences_per_stream, gpu_data.query_start_gpu + sequences_per_stream, gpu_data.query_end_gpu + sequences_per_stream,
                  gpu_data.scores_gpu + sequences_per_stream, openGap, extendGap, d_scoring_matrix, d_encoding_matrix);
            cudaErrchk(cudaGetLastError());

            asynch_mem_copies_dth(&gpu_data, alAbeg, alBbeg, top_scores_cpu, sequences_per_stream, sequences_stream_leftover, streams_cuda);

            alAbeg += stringsPerIt;
            alBbeg += stringsPerIt;
            alAend += stringsPerIt;
            alBend += stringsPerIt;
            top_scores_cpu += stringsPerIt;

        }  // for iterations end here

        cudaErrchk(cudaFree(strA_d));
        cudaErrchk(cudaFree(strB_d));
        cudaFreeHost(offsetA_h);
        cudaFreeHost(offsetB_h);

        for(int i = 0; i < NSTREAMS; i++)
          cudaStreamDestroy(streams_cuda[i]);

        std::cout <<"CPU time     = "<<std::fixed<<timer_cpu.getSeconds()    <<std::endl;
        std::cout <<"Packing time = "<<std::fixed<<timer_packing.getSeconds()<<std::endl;
        #pragma omp barrier
    }  // paralle pragma ends

    timer_total.stop();
    std::cout <<"Total Alignments   ="<<totalAlignments<<"\n"
              <<"Max Reference Size ="<<maxContigSize  <<"\n"
              <<"Max Query Size     ="<<maxReadSize    <<"\n"
              <<"Total Execution Time (seconds) = "<<timer_total.getSeconds() <<std::endl;
}

}

#endif
