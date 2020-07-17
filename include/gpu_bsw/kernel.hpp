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

enum class Direction {
  FORWARD,
  REVERSE
};

namespace gpu_bsw{

template<Direction DIR>
static __inline__ __device__ short
warpReduceMax_with_index(short val, short& myIndex, short& myIndex2, const unsigned lengthSeqB)
{
    constexpr int warpSize = 32;
    short myMax    = val;
    short newInd   = 0;
    short newInd2  = 0;
    short ind      = myIndex;
    short ind2     = myIndex2;
    unsigned mask  = __ballot_sync(0xffffffff, threadIdx.x < lengthSeqB);  // blockDim.x
    // unsigned newmask;
    for(int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        int tempVal = __shfl_down_sync(mask, val, offset);
        val     = max(val,tempVal);
        newInd  = __shfl_down_sync(mask, ind, offset);
        newInd2 = __shfl_down_sync(mask, ind2, offset);
        if(val != myMax)
        {
            ind   = newInd;
            ind2  = newInd2;
            myMax = val;
        }
        else if((val == tempVal) ) // this is kind of redundant and has been done purely to match the results
                                    // with SSW to get the smallest alignment with highest score. Theoreticaly
                                    // all the alignmnts with same score are same.
        {
            if((DIR==Direction::REVERSE && newInd2 > ind2) || (DIR==Direction::FORWARD && newInd < ind)){
              ind = newInd;
              ind2 = newInd2;
            }
        }
    }
    myIndex  = ind;
    myIndex2 = ind2;
    val      = myMax;
    return val;
}



template<Direction DIR>
static __device__ short
blockShuffleReduce_with_index(short myVal, short& myIndex, short& myIndex2, unsigned lengthSeqB)
{
    const int laneId = threadIdx.x % 32;
    const int warpId = threadIdx.x / 32;
    __shared__ short locTots[32];
    __shared__ short locInds[32];
    __shared__ short locInds2[32];
    short myInd  = myIndex;
    short myInd2 = myIndex2;
    myVal = warpReduceMax_with_index<DIR>(myVal, myInd, myInd2, lengthSeqB);

    __syncthreads();
    if(laneId == 0)
        locTots[warpId] = myVal;
    if(laneId == 0)
        locInds[warpId] = myInd;
    if(laneId == 0)
        locInds2[warpId] = myInd2;
    __syncthreads();
    unsigned check =
        ((32 + blockDim.x - 1) / 32);  // mimicing the ceil function for floats
                                       // float check = ((float)blockDim.x / 32);
    if(threadIdx.x < check)  /////******//////
    {
        myVal  = locTots[threadIdx.x];
        myInd  = locInds[threadIdx.x];
        myInd2 = locInds2[threadIdx.x];
    }
    else
    {
        myVal  = 0;
        myInd  = -1;
        myInd2 = -1;
    }
    __syncthreads();

    if(warpId == 0)
    {
        myVal    = warpReduceMax_with_index<DIR>(myVal, myInd, myInd2, lengthSeqB);
        myIndex  = myInd;
        myIndex2 = myInd2;
    }
    __syncthreads();
    return myVal;
}



__inline__ __device__ __host__ short
findMaxFour(const short first, const short second, const short third, const short fourth)
{
    short maxScore = 0;

    maxScore = max(first,second);
    maxScore = max(maxScore, third);
    maxScore = max(maxScore, fourth);

    return maxScore;
}



template<DataType DT, Direction DIR>
inline __global__ void
sequence_process(
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
){
  int block_Id  = blockIdx.x;
  int thread_Id = threadIdx.x;
  short laneId = threadIdx.x%32;
  short warpId = threadIdx.x/32;

  //Only used by DNA sequencing
  const short matchScore    = scoring_matrix[0];
  const short misMatchScore = scoring_matrix[1];

  unsigned lengthSeqA;
  unsigned lengthSeqB;

  // local pointers
  char*    seqA;
  char*    seqB;
  char* longer_seq;

  extern __shared__ char is_valid_array[];
  char*                  is_valid = &is_valid_array[0];

  // setting up block local sequences and their lengths.
  if(block_Id == 0)
  {
    seqA       = seqA_array;
    seqB       = seqB_array;
  }
  else
  {
    seqA       = seqA_array + prefix_lengthA[block_Id - 1];
    seqB       = seqB_array + prefix_lengthB[block_Id - 1];
  }

  if(DIR==Direction::FORWARD){
    if(block_Id == 0)
    {
      lengthSeqA = prefix_lengthA[0];
      lengthSeqB = prefix_lengthB[0];
    }
    else
    {
      lengthSeqA = prefix_lengthA[block_Id] - prefix_lengthA[block_Id - 1];
      lengthSeqB = prefix_lengthB[block_Id] - prefix_lengthB[block_Id - 1];
    }
  } else {
    lengthSeqA = seqA_align_end[block_Id];
    lengthSeqB = seqB_align_end[block_Id];
  }


  // what is the max length and what is the min length
  unsigned maxSize = lengthSeqA > lengthSeqB ? lengthSeqA : lengthSeqB;
  unsigned minSize = lengthSeqA < lengthSeqB ? lengthSeqA : lengthSeqB;

  // shared memory space for storing longer of the two strings
  memset(is_valid, 0, minSize);
  is_valid += minSize;
  memset(is_valid, 1, minSize);
  is_valid += minSize;
  memset(is_valid, 0, minSize);
  __syncthreads(); // this is required because shmem writes //TODO: May be unnecessary because shared memory isn't used for a while below and there's another synchronization point before it is.

  // When moving forward, the shorter of the two strings is stored in thread registers.

  // When moving in reverse, check if the new length of A is larger than B, if so then
  // place the B string in registers and A in myLocString, make sure we dont do redundant
  // copy by checking which string is located in myLocString before
  char myColumnChar;
  if(lengthSeqA < lengthSeqB)
  {
    const auto pos = (DIR==Direction::FORWARD) ? thread_Id : (lengthSeqA - 1) - thread_Id;
    if(thread_Id < lengthSeqA)
      myColumnChar = seqA[pos];  // read only once
    longer_seq = seqB;
  }
  else
  {
    const auto pos = (DIR==Direction::FORWARD) ? thread_Id : (lengthSeqB - 1) - thread_Id;
    if(thread_Id < lengthSeqB)
      myColumnChar = seqB[pos];
    longer_seq = seqA;
  }

  __syncthreads(); // this is required here so that complete sequence has been copied to shared memory

  int   i            = 1;
  short thread_max   = 0; // to maintain the thread max score
  short thread_max_i = 0; // to maintain the DP coordinate i for the longer string
  short thread_max_j = 0;// to maintain the DP cooirdinate j for the shorter string

  //initializing registers for storing diagonal values for three recent most diagonals (separate tables for
  //H, E and F)
  short _curr_H = 0, _curr_F = 0, _curr_E = 0;
  short _prev_H = 0, _prev_F = 0, _prev_E = 0;
  short _prev_prev_H = 0, _prev_prev_F = 0, _prev_prev_E = 0;
  short _temp_Val = 0;

  __shared__ short sh_prev_E[32]; // one such element is required per warp
  __shared__ short sh_prev_H[32];
  __shared__ short sh_prev_prev_H[32];

  __shared__ short local_spill_prev_E[1024];// each threads local spill,
  __shared__ short local_spill_prev_H[1024];
  __shared__ short local_spill_prev_prev_H[1024];

  //Used only by RNA. Has a length of 1 for DNA because length 0 is not allowed.
  __shared__ short sh_aa_encoding[(DT==DataType::RNA)?ENCOD_MAT_SIZE:1];// length = 91
  __shared__ short sh_aa_scoring [(DT==DataType::RNA)?SCORE_MAT_SIZE:1];

  if(DT==DataType::RNA){
    int max_threads = blockDim.x;
    for(int p = thread_Id; p < SCORE_MAT_SIZE; p+=max_threads){
      sh_aa_scoring[p] = scoring_matrix[p];
    }
    for(int p = thread_Id; p < ENCOD_MAT_SIZE; p+=max_threads){
      sh_aa_encoding[p] = encoding_matrix[p];
    }
  }

  __syncthreads(); // to make sure all shmem allocations have been initialized

  // iterate for the number of anti-diagonals
  for(int diag = 0; diag < lengthSeqA + lengthSeqB - 1; diag++)
  {

    is_valid = is_valid - (diag < minSize || diag >= maxSize); //move the pointer to left by 1 if cnd true

    // value exchange happens here to setup registers for next iteration
    _temp_Val = _prev_H;
    _prev_H = _curr_H;
    _curr_H = _prev_prev_H;
    _prev_prev_H = _temp_Val;
    _curr_H = 0;

    _temp_Val = _prev_E;
    _prev_E = _curr_E;
    _curr_E = _prev_prev_E;
    _prev_prev_E = _temp_Val;
    _curr_E = 0;

    _temp_Val = _prev_F;
    _prev_F = _curr_F;
    _curr_F = _prev_prev_F;
    _prev_prev_F = _temp_Val;
    _curr_F = 0;

    if(laneId == 31)
    { // if you are the last thread in your warp then spill your values to shmem
      sh_prev_E[warpId] = _prev_E;
      sh_prev_H[warpId] = _prev_H;
      sh_prev_prev_H[warpId] = _prev_prev_H;
    }

    if(diag >= maxSize)
    { // if you are invalid in this iteration, spill your values to shmem
      local_spill_prev_E[thread_Id] = _prev_E;
      local_spill_prev_H[thread_Id] = _prev_H;
      local_spill_prev_prev_H[thread_Id] = _prev_prev_H;
    }

    __syncthreads(); // this is needed so that all the shmem writes are completed.

    if(is_valid[thread_Id] && thread_Id < minSize)
    {
      unsigned mask  = __ballot_sync(__activemask(), (is_valid[thread_Id] &&( thread_Id < minSize)));
      short fVal = _prev_F + extendGap;
      short hfVal = _prev_H + startGap;
      short valeShfl = __shfl_sync(mask, _prev_E, laneId- 1, 32);
      short valheShfl = __shfl_sync(mask, _prev_H, laneId - 1, 32);
      short eVal=0;
      short heVal = 0;

      if(diag >= maxSize) // when the previous thread has phased out, get value from shmem
      {
        eVal = local_spill_prev_E[thread_Id - 1] + extendGap;
        heVal = local_spill_prev_H[thread_Id - 1]+ startGap;
      }
      else
      {
        eVal =((warpId !=0 && laneId == 0)?sh_prev_E[warpId-1]: valeShfl) + extendGap;
        heVal =((warpId !=0 && laneId == 0)?sh_prev_H[warpId-1]:valheShfl) + startGap;
      }

      if(warpId == 0 && laneId == 0) // make sure that values for lane 0 in warp 0 is not undefined
      {
        eVal = 0;
        heVal = 0;
      }
      _curr_F = (fVal > hfVal) ? fVal : hfVal;
      _curr_E = (eVal > heVal) ? eVal : heVal;

      short testShufll = __shfl_sync(mask, _prev_prev_H, laneId - 1, 32);
      short final_prev_prev_H = 0;

      if(diag >= maxSize)
      {
        final_prev_prev_H = local_spill_prev_prev_H[thread_Id - 1];
      }
      else
      {
        final_prev_prev_H =(warpId !=0 && laneId == 0)?sh_prev_prev_H[warpId-1]:testShufll;
      }

      if(warpId == 0 && laneId == 0) final_prev_prev_H = 0;

      const int diag_pos = (DIR==Direction::FORWARD) ? i-1 : maxSize-i;

      short diag_score;
      if(DT==DataType::DNA){
        diag_score = final_prev_prev_H + ((longer_seq[diag_pos] == myColumnChar) ? matchScore : misMatchScore);
      } else {
        short mat_index_q = sh_aa_encoding[(int)longer_seq[diag_pos]]; //encoding_matrix
        short mat_index_r = sh_aa_encoding[(int)myColumnChar];
        short add_score = sh_aa_scoring[mat_index_q*24 + mat_index_r]; // doesnt really matter in what order these indices are used, since the scoring table is symmetrical

        diag_score = final_prev_prev_H + add_score;
      }

      _curr_H = findMaxFour(diag_score, _curr_F, _curr_E, 0);
      if(DIR==Direction::FORWARD){
        thread_max_i = (thread_max >= _curr_H) ? thread_max_i : i;
        thread_max_j = (thread_max >= _curr_H) ? thread_max_j : thread_Id + 1;
      } else {
        thread_max_i = (thread_max >= _curr_H) ? thread_max_i : maxSize - i;            // begin_A (longer string)
        thread_max_j = (thread_max >= _curr_H) ? thread_max_j : minSize - thread_Id -1; // begin_B (shorter string)
      }
      thread_max   = (thread_max >= _curr_H) ? thread_max : _curr_H;
      i++;
    }
    __syncthreads(); // why do I need this? commenting it out breaks it
  }
  __syncthreads();

  thread_max = blockShuffleReduce_with_index<DIR>(thread_max, thread_max_i, thread_max_j, minSize);  // thread 0 will have the correct values

  if(thread_Id == 0)
  {
    if(DIR==Direction::FORWARD){
      if(lengthSeqA < lengthSeqB)
      {
        seqB_align_end[block_Id] = thread_max_i;
        seqA_align_end[block_Id] = thread_max_j;
        top_scores[block_Id] = thread_max;
      }
      else
      {
        seqA_align_end[block_Id] = thread_max_i;
        seqB_align_end[block_Id] = thread_max_j;
        top_scores[block_Id] = thread_max;
      }
    } else {
      if(lengthSeqA < lengthSeqB)
      {
        seqB_align_begin[block_Id] = thread_max_i; //newlengthSeqB
        seqA_align_begin[block_Id] = thread_max_j; //newlengthSeqA
      }
      else
      {
        seqA_align_begin[block_Id] = thread_max_i; //newlengthSeqA
        seqB_align_begin[block_Id] = thread_max_j; //newlengthSeqB
      }
    }
  }
  __syncthreads(); //TODO: May be unnecessary
}

}
#endif
