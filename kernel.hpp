#ifndef KERNEL_HPP
#define KERNEL_HPP
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#define NUM_OF_AA 21
#define ENCOD_MAT_SIZE 91
#define SCORE_MAT_SIZE 576

enum class DataType {
  DNA,
  RNA
};

namespace gpu_bsw{
__device__ short
warpReduceMax_with_index(short val, short& myIndex, short& myIndex2, unsigned lengthSeqB);

__device__ short
warpReduceMax_with_index_reverse(short val, short& myIndex, short& myIndex2, unsigned lengthSeqB);

__device__ short
warpReduceMax(short val, unsigned lengthSeqB);

__device__ short
blockShuffleReduce_with_index(short myVal, short& myIndex, short& myIndex2, unsigned lengthSeqB);

__device__ short
blockShuffleReduce_with_index_reverse(short myVal, short& myIndex, short& myIndex2, unsigned lengthSeqB);

__device__ short
blockShuffleReduce(short val, unsigned lengthSeqB);

__device__ __host__ short
findMax(short array[], int length, int* ind);

__device__ __host__ short
findMaxFour(short first, short second, short third, short fourth);

__global__ void
sequence_dna_kernel(
  char       *const seqA_array,
  char       *const seqB_array,
  unsigned   *const prefix_lengthA,
  unsigned   *const prefix_lengthB,
  short      *const seqA_align_begin,
  short      *const seqA_align_end,
  short      *const seqB_align_begin,
  short      *const seqB_align_end,
  short      *const top_scores,
  const short       matchScore,
  const short       misMatchScore,
  const short       startGap,
  const short       extendGap
);

__global__ void
sequence_dna_reverse(
  char       *const seqA_array,
  char       *const seqB_array,
  unsigned   *const prefix_lengthA,
  unsigned   *const prefix_lengthB,
  short      *const seqA_align_begin,
  short      *const seqA_align_end,
  short      *const seqB_align_begin,
  short      *const seqB_align_end,
  short      *const top_scores,
  const short       matchScore,
  const short       misMatchScore,
  const short       startGap,
  const short       extendGap
);

__global__ void
sequence_aa_kernel(
  char        *const seqA_array,
  char        *const seqB_array,
  unsigned    *const prefix_lengthA,
  unsigned    *const prefix_lengthB,
  short       *const seqA_align_begin,
  short       *const seqA_align_end,
  short       *const seqB_align_begin,
  short       *const seqB_align_end,
  short       *const top_scores,
  const short        startGap,
  const short        extendGap,
  const short *const scoring_matrix,
  const short *const encoding_matrix
);

__global__ void
sequence_aa_reverse(
  char        *const seqA_array,
  char        *const seqB_array,
  unsigned    *const prefix_lengthA,
  unsigned    *const prefix_lengthB,
  short       *const seqA_align_begin,
  short       *const seqA_align_end,
  short       *const seqB_align_begin,
  short       *const seqB_align_end,
  short       *const top_scores,
  const short        startGap,
  const short        extendGap,
  const short *const scoring_matrix,
  const short *const encoding_matrix
);

}
#endif
