#ifndef KERNEL_HPP
#define KERNEL_HPP
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sys/time.h>

#define EXTEND_GAP -2
#define START_GAP -2
//#define NBLOCKS 15000
#define MATCH 15
#define MISMATCH -3

__inline__ __device__ short
warpReduceMax(short val, short& myIndex, short& myIndex2);

__device__ short
blockShuffleReduce(short myVal, short& myIndex, short& myIndex2);

__device__ __host__ short
           findMax(short array[], int length, int* ind);

__device__ __host__ short
           findMax(short array[], int length, int* ind);

__device__ void
traceBack(short current_i, short current_j, short* seqA_align_begin,
          short* seqB_align_begin, const char* seqA, const char* seqB, short* I_i,
          short* I_j, unsigned lengthSeqB, unsigned lengthSeqA, unsigned int* diagOffset);

__global__ void
align_sequences_gpu(char* seqA_array, char* seqB_array, unsigned* prefix_lengthA,
                    unsigned* prefix_lengthB, unsigned maxMatrixSize, short* I_i_array,
                    short* I_j_array, short* seqA_align_begin, short* seqA_align_end,
                    short* seqB_align_begin, short* seqB_align_end, short* top_scores);

#endif
