
#include <kernel.hpp>

__inline__ __device__ short
warpReduceMax(short val, short& myIndex, short& myIndex2)
{
    int   warpSize = 32;
    short myMax    = 0;
    short newInd   = 0;
    short newInd2  = 0;
    short ind      = myIndex;
    short ind2     = myIndex2;

    myMax         = val;
    unsigned mask = __ballot_sync(0xffffffff, threadIdx.x < blockDim.x);

    for(int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        val     = max(val, __shfl_down_sync(mask, val, offset));
        newInd  = __shfl_down_sync(mask, ind, offset);
        newInd2 = __shfl_down_sync(mask, ind2, offset);

        if(val != myMax)
        {
            ind   = newInd;
            ind2  = newInd2;
            myMax = val;
        }
        __syncthreads();
    }
    __syncthreads();
    myIndex  = ind;
    myIndex2 = ind2;
    val      = myMax;
    return val;
}

__device__ short
blockShuffleReduce(short myVal, short& myIndex, short& myIndex2)
{
    int              laneId = threadIdx.x % 32;
    int              warpId = threadIdx.x / 32;
    __shared__ short locTots[32];
    __shared__ short locInds[32];
    __shared__ short locInds2[32];
    short            myInd  = myIndex;
    short            myInd2 = myIndex2;
    myVal                   = warpReduceMax(myVal, myInd, myInd2);

    if(laneId == 0)
        locTots[warpId] = myVal;
    if(laneId == 0)
        locInds[warpId] = myInd;
    if(laneId == 0)
        locInds2[warpId] = myInd2;

    __syncthreads();

    if(threadIdx.x <= (blockDim.x / 32))
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
        myVal    = warpReduceMax(myVal, myInd, myInd2);
        myIndex  = myInd;
        myIndex2 = myInd2;
    }
    return myVal;
}
__device__ __host__ short
           findMax(short array[], int length, int* ind)
{
    short max = array[0];
    *ind      = 0;

    for(int i = 1; i < length; i++)
    {
        if(array[i] > max)
        {
            max  = array[i];
            *ind = i;
        }
    }
    return max;
}

__device__ void
traceBack(short current_i, short current_j, short* seqA_align_begin,
          short* seqB_align_begin, const char* seqA, const char* seqB, short* I_i,
          short* I_j, unsigned lengthSeqB, unsigned lengthSeqA, unsigned int* diagOffset)
{
    // int current_i=i_max,current_j=j_max;
    // printf ("SeqA:%d SeqB:%d\n",lengthSeqA,lengthSeqB);
    // if(blockIdx.x == 265)
    //	for(int i = lengthSeqA+lengthSeqB; i>lengthSeqA+lengthSeqB -10; i--){
    //		printf ("diagOffset:%d \n",diagOffset[i]);
    //	}
    int            myId = blockIdx.x;
    unsigned short current_diagId;     // = current_i+current_j;
    unsigned short current_locOffset;  // = 0;

    current_diagId    = current_i + current_j;
    current_locOffset = 0;
    if(current_diagId < lengthSeqA + 1)
    {
        current_locOffset = current_j;
    }
    else
    {
        unsigned short myOff = current_diagId - lengthSeqA;
        current_locOffset    = current_j - myOff;
    }
    //	I_i[diagOffset[diagId]+locOffset] = i + iVal[ind]; // coalesced accesses,
    //	I_j[diagOffset[diagId]+locOffset] = j + jVal[ind]; // coalesced accesses,
    // printf ("current_i:%d current_j:%d\n",current_i,current_j);
    // if(blockIdx.x==265)
    // printf ("diagOffset:%d
    // current_locOffset:%d\n",diagOffset[current_diagId],current_locOffset);

    short next_i = I_i[diagOffset[current_diagId] + current_locOffset];
    short next_j = I_j[diagOffset[current_diagId] + current_locOffset];
    // printf ("next_i:%d next_j:%d\n",next_i,next_j);
    // printf ("diagOffset:%d
    // current_locOffset:%d\n",diagOffset[current_diagId],current_locOffset);

    while(((current_i != next_i) || (current_j != next_j)) && (next_j != 0) &&
          (next_i != 0))
    {
        current_i = next_i;
        current_j = next_j;

        current_diagId    = current_i + current_j;
        current_locOffset = 0;
        if(current_diagId < lengthSeqA + 1)
        {
            current_locOffset = current_j;
        }
        else
        {
            unsigned short myOff2 = current_diagId - lengthSeqA;
            current_locOffset     = current_j - myOff2;
        }

        next_i = I_i[diagOffset[current_diagId] + current_locOffset];
        next_j = I_j[diagOffset[current_diagId] + current_locOffset];
    }
    // printf("final current_i=%d, current_j=%d\n", current_i, current_j);
    seqA_align_begin[myId] = current_i;
    seqB_align_begin[myId] = current_j;
    // printf("traceback done\n");

    // *tick_out = tick;
}

__global__ void
align_sequences_gpu(char* seqA_array, char* seqB_array, unsigned* prefix_lengthA,
                    unsigned* prefix_lengthB, unsigned* prefix_matrices, short* I_i_array,
                    short* I_j_array, short* seqA_align_begin, short* seqA_align_end,
                    short* seqB_align_begin, short* seqB_align_end)
{
    int myId  = blockIdx.x;
    int myTId = threadIdx.x;

    //	if(myId == 4)
    //	printf("block3 lenA:%d",prefix_lengthA[myId]);
    unsigned lengthSeqA;
    unsigned lengthSeqB;
    unsigned matrixOffset;
    // local pointers
    char*    seqA;
    char*    seqB;
    short *  I_i, *I_j;
    unsigned totBytes = 0;

    extern __shared__ char is_valid_array[];
    char*                  is_valid = &is_valid_array[0];

    if(myId == 0)
    {
        lengthSeqA   = prefix_lengthA[0];
        lengthSeqB   = prefix_lengthB[0];
        matrixOffset = 0;
        seqA         = seqA_array;
        seqB         = seqB_array;
        I_i          = I_i_array;
        I_j          = I_j_array;
    }
    else
    {
        lengthSeqA   = prefix_lengthA[myId] - prefix_lengthA[myId - 1];
        lengthSeqB   = prefix_lengthB[myId] - prefix_lengthB[myId - 1];
        matrixOffset = prefix_matrices[myId - 1];
        seqA         = seqA_array + prefix_lengthA[myId - 1];
        seqB         = seqB_array + prefix_lengthB[myId - 1];
        I_i          = I_i_array + matrixOffset;
        I_j          = I_j_array + matrixOffset;
    }

    short* curr_H =
        (short*) (&is_valid_array[3 * lengthSeqB +
                                  (lengthSeqB & 1)]);  // point where the valid_array ends
    short* prev_H      = &curr_H[lengthSeqB + 1];      // where the curr_H array ends
    short* prev_prev_H = &prev_H[lengthSeqB + 1];
    totBytes += (3 * lengthSeqB + (lengthSeqB & 1)) * sizeof(char) +
                ((lengthSeqB + 1) + (lengthSeqB + 1)) * sizeof(short);

    short* curr_E      = &prev_prev_H[lengthSeqB + 1];
    short* prev_E      = &curr_E[lengthSeqB + 1];
    short* prev_prev_E = &prev_E[lengthSeqB + 1];
    totBytes += ((lengthSeqB + 1) + (lengthSeqB + 1) + (lengthSeqB + 1)) * sizeof(short);

    short* curr_F      = &prev_prev_E[lengthSeqB + 1];
    short* prev_F      = &curr_F[lengthSeqB + 1];
    short* prev_prev_F = &prev_F[lengthSeqB + 1];
    totBytes += ((lengthSeqB + 1) + (lengthSeqB + 1) + (lengthSeqB + 1)) * sizeof(short);

    char* myLocString = (char*) &prev_prev_F[lengthSeqB + 1];
    totBytes += (lengthSeqB + 1) * sizeof(short) + (lengthSeqA) * sizeof(char);

    // if(myTId == 0 && myId == 0)
    // printf("outalign: %d locAlign: %d", outalign, totBytes);
    unsigned      alignmentPad = 4 - totBytes % 4;
    unsigned int* diagOffset   = (unsigned int*) &myLocString[lengthSeqA + alignmentPad];
    // char* v = is_valid;

    __syncthreads();
    // if(myTId == 0) {
    memset(is_valid, 0, lengthSeqB);
    is_valid += lengthSeqB;
    memset(is_valid, 1, lengthSeqB);
    is_valid += lengthSeqB;
    memset(is_valid, 0, lengthSeqB);

    memset(curr_H, 0, 9 * (lengthSeqB + 1) * sizeof(short));
    //}
    //__shared__ int global_max;
    __shared__ int i_max;
    __shared__ int j_max;
    int            j            = myTId + 1;
    char           myColumnChar = seqB[j - 1];  // read only once

    ///////////locsl dtring read in
    for(int i = myTId; i < lengthSeqA; i += 32)
    {
        myLocString[i] = seqA[i];
    }

    __syncthreads();

    short            traceback[4];
    __shared__ short iVal[4];  //= {-1,-1,0,0};
    iVal[0] = -1;
    iVal[1] = -1;
    iVal[2] = 0;
    iVal[3] = 0;
    __shared__ short jVal[4];  //= {-1,0,-1,0};
    jVal[0] = -1;
    jVal[1] = 0;
    jVal[2] = -1;
    jVal[3] = 0;

    //__shared__ unsigned int diagOffset[1071+128+2]; //make this dynamic shared memory
    // later on
    int ind;

    int   i            = 1;
    short thread_max   = 0;
    short thread_max_i = 0;
    short thread_max_j = 0;

    short* tmp_ptr;
    int    locSum = 0;
    for(int diag = 0; diag < lengthSeqA + lengthSeqB - 1; diag++)
    {
        int locDiagId = diag + 1;
        if(myTId == 0)
        {  // this computes the prefixSum for diagonal offset look up table.
            if(locDiagId <= lengthSeqB + 1)
            {
                locSum += locDiagId;
                diagOffset[locDiagId] = locSum;
            }
            else if(locDiagId > lengthSeqA + 1)
            {
                locSum += (lengthSeqB + 1) - (locDiagId - (lengthSeqA + 1));
                diagOffset[locDiagId] = locSum;
            }
            else
            {
                locSum += lengthSeqB + 1;
                diagOffset[locDiagId] = locSum;
            }
            diagOffset[lengthSeqA + lengthSeqB] = locSum + 2;
            // printf("diag:%d\tlocSum:%d\n",diag,diagOffset[locDiagId]);
        }
    }
    __syncthreads();
    for(int diag = 0; diag < lengthSeqA + lengthSeqB - 1; diag++)
    {  // iterate for the number of anti-diagonals

        is_valid = is_valid - (diag < lengthSeqB || diag >= lengthSeqA);

        tmp_ptr     = prev_H;
        prev_H      = curr_H;
        curr_H      = prev_prev_H;
        prev_prev_H = tmp_ptr;

        memset(curr_H, 0, (lengthSeqB + 1) * sizeof(short));
        __syncthreads();
        tmp_ptr     = prev_E;
        prev_E      = curr_E;
        curr_E      = prev_prev_E;
        prev_prev_E = tmp_ptr;

        memset(curr_E, 0, (lengthSeqB + 1) * sizeof(short));
        __syncthreads();
        tmp_ptr     = prev_F;
        prev_F      = curr_F;
        curr_F      = prev_prev_F;
        prev_prev_F = tmp_ptr;

        memset(curr_F, 0, (lengthSeqB + 1) * sizeof(short));

        __syncthreads();

        if(is_valid[myTId])
        {
            short fVal  = prev_F[j] + EXTEND_GAP;
            short hfVal = prev_H[j] + START_GAP;
            short eVal  = prev_E[j - 1] + EXTEND_GAP;
            short heVal = prev_H[j - 1] + START_GAP;

            curr_F[j] = (fVal > hfVal) ? fVal : hfVal;
            curr_E[j] = (eVal > heVal) ? eVal : heVal;

            //(myLocString[i-1] == myColumnChar)?MATCH:MISMATCH

            traceback[0] =
                prev_prev_H[j - 1] +
                ((myLocString[i - 1] == myColumnChar)
                     ? MATCH
                     : MISMATCH);  // similarityScore(myLocString[i-1],myColumnChar);//seqB[j-1]
            traceback[1] = curr_F[j];
            traceback[2] = curr_E[j];
            traceback[3] = 0;

            curr_H[j] = findMax(traceback, 4, &ind);
            //	I_i[i*(lengthSeqB+1)+j] = i + iVal[ind]; // non-coalesced accesses, need
            // to change
            //		I_j[i*(lengthSeqB+1)+j] = j + jVal[ind]; // non-coalesced accesses,
            //need to change

            /////////////////// coalesced version
            ///////////////
            unsigned short diagId    = i + j;
            unsigned short locOffset = 0;
            if(diagId < lengthSeqA + 1)
            {
                locOffset = j;
            }
            else
            {
                unsigned short myOff = diagId - lengthSeqA;
                locOffset            = j - myOff;
            }
            I_i[diagOffset[diagId] + locOffset] =
                i + iVal[ind];  // coalesced accesses, need to change
            I_j[diagOffset[diagId] + locOffset] =
                j + jVal[ind];  // coalesced accesses, need to change
                                /////////////
                                ///////////

            thread_max_i = (thread_max >= curr_H[j]) ? thread_max_i : i;
            thread_max_j = (thread_max >= curr_H[j]) ? thread_max_j : myTId + 1;
            thread_max   = (thread_max >= curr_H[j]) ? thread_max : curr_H[j];

            i++;
        }
    }

    // atomicMax(&global_max, thread_max);
    thread_max = blockShuffleReduce(
        thread_max, thread_max_i, thread_max_j);  // thread 0 will have the correct values
    // if(myId == 0 && myTId == 0)
    // printf("thread_max:%d thread_i:%d thread_j:%d \n",thread_max, thread_max_i,
    // thread_max_j);

    // traceback
    if(myTId == 0)
    {
        i_max           = thread_max_i;
        j_max           = thread_max_j;
        short current_i = i_max, current_j = j_max;
        seqA_align_end[myId] = current_i;
        seqB_align_end[myId] = current_j;
        //		for(int i = lengthSeqA+lengthSeqB; i>lengthSeqA+lengthSeqB -10; i--){
        //		printf ("diagOffset:%d \n",diagOffset[i]);
        //	}
        traceBack(current_i, current_j, seqA_align_begin, seqB_align_begin, seqA, seqB,
                  I_i, I_j, lengthSeqB, lengthSeqA, diagOffset);
    }
    __syncthreads();
}
