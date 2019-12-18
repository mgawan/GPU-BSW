/****************************

GPU accelerated Smith-Waterman for performing batch alignments (GPU-BSW) Copyright (c) 2019, The
Regents of the University of California, through Lawrence Berkeley National
Laboratory (subject to receipt of any required approvals from the U.S.
Dept. of Energy).  All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative
works, and perform publicly and display publicly, and to permit other to do
so.

****************************/


#ifndef KERNEL_HPP
#define KERNEL_HPP
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sys/time.h>

#define EXTEND_GAP -2
#define START_GAP -5
//#define NBLOCKS 15000
#define MATCH 2
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
                    unsigned* prefix_lengthB, short* seqA_align_begin,
                    short* seqA_align_end, short* seqB_align_begin,
                    short* seqB_align_end);

#endif
