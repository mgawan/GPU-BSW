
#include <kernel.hpp>

__inline__ __device__ short
warpReduceMax(short val, short& myIndex, short& myIndex2, unsigned lengthSeqB)
{
    int   warpSize = 32;
    short myMax    = 0;
    short newInd   = 0;
    short newInd2  = 0;
    short ind      = myIndex;
    short ind2     = myIndex2;
    myMax          = val;
    unsigned mask  = __ballot_sync(0xffffffff, threadIdx.x < lengthSeqB);  // blockDim.x
    // unsigned newmask;
    for(int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        int tempVal = __shfl_down_sync(mask, val, offset);
        val         = max(val, tempVal);
        newInd      = __shfl_down_sync(mask, ind, offset);
        newInd2     = __shfl_down_sync(mask, ind2, offset);

        if(val != myMax)
        {
            ind   = newInd;
            ind2  = newInd2;
            myMax = val;
        }
        else if((val == tempVal))
        {
            if(newInd < ind)
            {
                ind  = newInd;
                ind2 = newInd2;
            }
        }
    }
    myIndex  = ind;
    myIndex2 = ind2;
    val      = myMax;
    return val;
}

__device__ short
blockShuffleReduce(short myVal, short& myIndex, short& myIndex2, unsigned lengthSeqB)
{
    int              laneId = threadIdx.x % 32;
    int              warpId = threadIdx.x / 32;
    __shared__ short locTots[32];
    __shared__ short locInds[32];
    __shared__ short locInds2[32];
    short            myInd  = myIndex;
    short            myInd2 = myIndex2;
    myVal                   = warpReduceMax(myVal, myInd, myInd2, lengthSeqB);

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
        myVal    = warpReduceMax(myVal, myInd, myInd2, lengthSeqB);
        myIndex  = myInd;
        myIndex2 = myInd2;
    }
    __syncthreads();
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
    int            myId = blockIdx.x;
    unsigned short current_diagId;     // = current_i+current_j;
    unsigned short current_locOffset;  // = 0;
    unsigned       maxSize = lengthSeqA > lengthSeqB ? lengthSeqA : lengthSeqB;

    current_diagId    = current_i + current_j;
    current_locOffset = 0;
    if(current_diagId < maxSize + 1)
    {
        current_locOffset = current_j;
    }
    else
    {
        unsigned short myOff = current_diagId - maxSize;
        current_locOffset    = current_j - myOff;
    }

    short next_i = I_i[diagOffset[current_diagId] + current_locOffset];
    short next_j = I_j[diagOffset[current_diagId] + current_locOffset];

    while(((current_i != next_i) || (current_j != next_j)) && (next_j != 0) &&
          (next_i != 0))
    {
        current_i = next_i;
        current_j = next_j;

        current_diagId    = current_i + current_j;
        current_locOffset = 0;
        if(current_diagId < maxSize + 1)
        {
            current_locOffset = current_j;
        }
        else
        {
            unsigned short myOff2 = current_diagId - maxSize;
            current_locOffset     = current_j - myOff2;
        }

        next_i = I_i[diagOffset[current_diagId] + current_locOffset];
        next_j = I_j[diagOffset[current_diagId] + current_locOffset];
    }
    // printf("final current_i=%d, current_j=%d\n", current_i, current_j);
    if(lengthSeqA < lengthSeqB)
    {
        seqB_align_begin[myId] = current_i;
        seqA_align_begin[myId] = current_j;
    }
    else
    {
        seqA_align_begin[myId] = current_i;
        seqB_align_begin[myId] = current_j;
    }

    // printf("traceback done\n");

    // *tick_out = tick;
}

__global__ void
align_sequences_gpu(char* seqA_array, char* seqB_array, unsigned* prefix_lengthA,
                    unsigned* prefix_lengthB, short* seqA_align_begin,
                    short* seqA_align_end, short* seqB_align_begin, short* seqB_align_end)
{
    int      myId  = blockIdx.x;
    int      myTId = threadIdx.x;
    unsigned lengthSeqA;
    unsigned lengthSeqB;
    // local pointers
    char*    seqA;
    char*    seqB;
    unsigned totBytes = 0;

    extern __shared__ char is_valid_array[];
    char*                  is_valid = &is_valid_array[0];

    if(myId == 0)
    {
        lengthSeqA = prefix_lengthA[0];
        lengthSeqB = prefix_lengthB[0];
        seqA       = seqA_array;
        seqB       = seqB_array;
    }
    else
    {
        lengthSeqA = prefix_lengthA[myId] - prefix_lengthA[myId - 1];
        lengthSeqB = prefix_lengthB[myId] - prefix_lengthB[myId - 1];
        seqA       = seqA_array + prefix_lengthA[myId - 1];
        seqB       = seqB_array + prefix_lengthB[myId - 1];
    }

    unsigned maxSize = lengthSeqA > lengthSeqB ? lengthSeqA : lengthSeqB;
    unsigned minSize = lengthSeqA < lengthSeqB ? lengthSeqA : lengthSeqB;

    char* myLocString = (char*) &is_valid[3 * minSize + (minSize & 1)];

    __syncthreads();
    memset(is_valid, 0, minSize);
    is_valid += minSize;
    memset(is_valid, 1, minSize);
    is_valid += minSize;
    memset(is_valid, 0, minSize);

    int  j = myTId + 1;
    char myColumnChar;
    if(lengthSeqA < lengthSeqB)
    {
        myColumnChar = seqA[j - 1];  // read only once
        for(int i = myTId; i < lengthSeqB; i += 32)
        {
            myLocString[i] = seqB[i];
        }
    }
    else
    {
        myColumnChar = seqB[j - 1];
        for(int i = myTId; i < lengthSeqA; i += 32)
        {
            myLocString[i] = seqA[i];
        }
    }

    __syncthreads();

    short traceback[4];
    int   ind;
    int   i            = 1;
    short thread_max   = 0;
    short thread_max_i = 0;
    short thread_max_j = 0;

    //  short* tmp_ptr;
    short            _curr_H = 0, _curr_F = 0, _curr_E = 0;
    short            _prev_H = 0, _prev_F = 0, _prev_E = 0;
    short            _prev_prev_H = 0, _prev_prev_F = 0, _prev_prev_E = 0;
    short            _temp_Val = 0;
    __shared__ short sh_prev_E[12];
    __shared__ short sh_prev_H[12];
    __shared__ short sh_prev_prev_H[12];
    __shared__ short sh_last_prev_E, sh_last_prev_H, sh_last_prev_prev_H;
    sh_prev_E[0]      = 0;
    sh_prev_E[1]      = 0;
    sh_prev_H[0]      = 0;
    sh_prev_H[1]      = 0;
    sh_prev_prev_H[0] = 0;
    sh_prev_prev_H[1] = 0;

    __syncthreads();
    for(int diag = 0; diag < lengthSeqA + lengthSeqB - 1; diag++)
    {  // iterate for the number of anti-diagonals

        is_valid = is_valid - (diag < minSize || diag >= maxSize);

        _temp_Val    = _prev_H;
        _prev_H      = _curr_H;
        _curr_H      = _prev_prev_H;
        _prev_prev_H = _temp_Val;

        _curr_H = 0;

        __syncthreads();

        _temp_Val    = _prev_E;
        _prev_E      = _curr_E;
        _curr_E      = _prev_prev_E;
        _prev_prev_E = _temp_Val;

        _curr_E = 0;
        __syncthreads();

        _temp_Val    = _prev_F;
        _prev_F      = _curr_F;
        _curr_F      = _prev_prev_F;
        _prev_prev_F = _temp_Val;
        // memset(curr_F, 0, (minSize + 1) * sizeof(short));
        _curr_F      = 0;
        short laneId = threadIdx.x % 32;
        short warpId = threadIdx.x / 32;
        if(laneId == 31)
        {
            sh_prev_E[warpId]      = _prev_E;
            sh_prev_H[warpId]      = _prev_H;
            sh_prev_prev_H[warpId] = _prev_prev_H;
        }

        if(threadIdx.x == blockDim.x - 2)
        {
            sh_last_prev_E      = _prev_E;
            sh_last_prev_H      = _prev_H;
            sh_last_prev_prev_H = _prev_prev_H;
        }
        __syncthreads();

        if(is_valid[myTId] && myTId < minSize)
        {
            unsigned mask =
                __ballot_sync(__activemask(), (is_valid[myTId] && (myTId < minSize)));

            short fVal  = _prev_F + EXTEND_GAP;
            short hfVal = _prev_H + START_GAP;

            short valeShfl  = __shfl_sync(mask, _prev_E, laneId - 1, 32);
            short valheShfl = __shfl_sync(mask, _prev_H, laneId - 1, 32);

            short eVal =
                ((warpId != 0 && laneId == 0) ? sh_prev_E[warpId - 1] : valeShfl) +
                EXTEND_GAP;
            short heVal =
                ((warpId != 0 && laneId == 0) ? sh_prev_H[warpId - 1] : valheShfl) +
                START_GAP;
            if(warpId == 0 && laneId == 0)
            {
                eVal  = 0 + EXTEND_GAP;
                heVal = 0 + START_GAP;
            }
            if(threadIdx.x == blockDim.x - 1)
            {
                eVal  = sh_last_prev_E + EXTEND_GAP;
                heVal = sh_last_prev_H + START_GAP;
            }

            _curr_F = (fVal > hfVal) ? fVal : hfVal;
            _curr_E = (eVal > heVal) ? eVal : heVal;

            short testShufll = __shfl_sync(mask, _prev_prev_H, laneId - 1, 32);
            short valPPh =
                (warpId != 0 && laneId == 0) ? sh_prev_prev_H[warpId - 1] : testShufll;
            if(warpId == 0 && laneId == 0)
                valPPh = 0;
            if(threadIdx.x == blockDim.x - 1)
            {
                valPPh = sh_last_prev_prev_H;  // = _prev_prev_H;
            }

            traceback[0] =
                valPPh +
                ((myLocString[i - 1] == myColumnChar)
                     ? MATCH
                     : MISMATCH);  // similarityScore(myLocString[i-1],myColumnChar);//seqB[j-1]
            traceback[1] = _curr_F;
            traceback[2] = _curr_E;
            traceback[3] = 0;

            _curr_H = findMax(traceback, 4, &ind);

            thread_max_i = (thread_max >= _curr_H) ? thread_max_i : i;
            thread_max_j = (thread_max >= _curr_H) ? thread_max_j : myTId + 1;
            thread_max   = (thread_max >= _curr_H) ? thread_max : _curr_H;

            i++;
        }

        __syncthreads();
    }
    __syncthreads();

    thread_max = blockShuffleReduce(thread_max, thread_max_i, thread_max_j,
                                    minSize);  // thread 0 will have the correct values

    __syncthreads();

    if(myTId == 0)
    {
        if(lengthSeqA < lengthSeqB)
        {
            seqB_align_end[myId] = thread_max_i;
            seqA_align_end[myId] = thread_max_j;
        }
        else
        {
            seqA_align_end[myId] = thread_max_i;
            seqB_align_end[myId] = thread_max_j;
        }
    }

    __syncthreads();

    j = myTId + 1;

    int newlengthSeqA = seqA_align_end[myId];
    int newlengthSeqB = seqB_align_end[myId];
    int seqAend       = newlengthSeqA - 1;
    int seqBend       = newlengthSeqB - 1;
    __syncthreads();

    if(lengthSeqA < lengthSeqB)
    {
        if(myTId < newlengthSeqA)
            seqA[seqAend - (j - 1)] = myColumnChar;

        for(int i = myTId; i < newlengthSeqB; i += 32)
        {
            seqB[seqBend - i] = myLocString[i];
        }
    }
    else
    {
        if(myTId < newlengthSeqB)
            seqB[seqBend - (j - 1)] = myColumnChar;

        for(int i = myTId; i < newlengthSeqA; i += 32)
        {
            seqA[seqAend - i] = myLocString[i];
        }
    }

    __syncthreads();

    maxSize = newlengthSeqA > newlengthSeqB ? newlengthSeqA : newlengthSeqB;
    minSize = newlengthSeqA < newlengthSeqB ? newlengthSeqA : newlengthSeqB;
    __syncthreads();

    is_valid = &is_valid_array[0];  // reset is_valid array for second iter
    memset(is_valid, 0, minSize);
    is_valid += minSize;
    memset(is_valid, 1, minSize);
    is_valid += minSize;
    memset(is_valid, 0, minSize);
    __syncthreads();

    if(lengthSeqA < lengthSeqB)
    {
        if(j < newlengthSeqA)
            ;
        myColumnChar = seqA[j - 1];  // read only once
        for(int i = myTId; i < newlengthSeqB; i += 32)
        {
            myLocString[i] = seqB[i];  // locString contains reference/longer string
        }
    }
    else
    {
        if(j < newlengthSeqB)
            ;
        myColumnChar = seqB[j - 1];
        for(int i = myTId; i < newlengthSeqA; i += 32)
        {
            myLocString[i] = seqA[i];
        }
    }

    __syncthreads();
    i            = 1;
    thread_max   = 0;
    thread_max_i = 0;
    thread_max_j = 0;

    _curr_H = 0, _curr_F = 0, _curr_E = 0;
    _prev_H = 0, _prev_F = 0, _prev_E = 0;
    _prev_prev_H = 0, _prev_prev_F = 0, _prev_prev_E = 0;

    sh_prev_E[0]      = 0;
    sh_prev_E[1]      = 0;
    sh_prev_H[0]      = 0;
    sh_prev_H[1]      = 0;
    sh_prev_prev_H[0] = 0;
    sh_prev_prev_H[1] = 0;

    __syncthreads();
    for(int diag = 0; diag < newlengthSeqA + newlengthSeqB - 1; diag++)
    {  // iterate for the number of anti-diagonals

        is_valid = is_valid - (diag < minSize || diag >= maxSize);

        _temp_Val    = _prev_H;
        _prev_H      = _curr_H;
        _curr_H      = _prev_prev_H;
        _prev_prev_H = _temp_Val;

        _curr_H = 0;

        __syncthreads();

        _temp_Val    = _prev_E;
        _prev_E      = _curr_E;
        _curr_E      = _prev_prev_E;
        _prev_prev_E = _temp_Val;
        _curr_E      = 0;
        __syncthreads();

        _temp_Val    = _prev_F;
        _prev_F      = _curr_F;
        _curr_F      = _prev_prev_F;
        _prev_prev_F = _temp_Val;
        _curr_F      = 0;
        short laneId = threadIdx.x % 32;
        short warpId = threadIdx.x / 32;
        if(laneId == 31)
        {
            sh_prev_E[warpId]      = _prev_E;
            sh_prev_H[warpId]      = _prev_H;
            sh_prev_prev_H[warpId] = _prev_prev_H;
        }

        if(threadIdx.x == blockDim.x - 2)
        {
            sh_last_prev_E      = _prev_E;
            sh_last_prev_H      = _prev_H;
            sh_last_prev_prev_H = _prev_prev_H;
        }
        __syncthreads();

        if(is_valid[myTId] && myTId < minSize)
        {
            unsigned mask =
                __ballot_sync(__activemask(), (is_valid[myTId] && (myTId < minSize)));

            short fVal  = _prev_F + EXTEND_GAP;
            short hfVal = _prev_H + START_GAP;

            short valeShfl  = __shfl_sync(mask, _prev_E, laneId - 1, 32);
            short valheShfl = __shfl_sync(mask, _prev_H, laneId - 1, 32);

            short eVal =
                ((warpId != 0 && laneId == 0) ? sh_prev_E[warpId - 1] : valeShfl) +
                EXTEND_GAP;
            short heVal =
                ((warpId != 0 && laneId == 0) ? sh_prev_H[warpId - 1] : valheShfl) +
                START_GAP;
            if(warpId == 0 && laneId == 0)
            {
                eVal  = 0 + EXTEND_GAP;
                heVal = 0 + START_GAP;
            }
            if(threadIdx.x == blockDim.x - 1)
            {
                eVal  = sh_last_prev_E + EXTEND_GAP;
                heVal = sh_last_prev_H + START_GAP;
            }

            _curr_F = (fVal > hfVal) ? fVal : hfVal;
            _curr_E = (eVal > heVal) ? eVal : heVal;

            short testShufll = __shfl_sync(mask, _prev_prev_H, laneId - 1, 32);
            short valPPh =
                (warpId != 0 && laneId == 0) ? sh_prev_prev_H[warpId - 1] : testShufll;
            if(warpId == 0 && laneId == 0)
                valPPh = 0;
            if(threadIdx.x == blockDim.x - 1)
            {
                valPPh = sh_last_prev_prev_H;  // = _prev_prev_H;
            }

            traceback[0] =
                valPPh +
                ((myLocString[i - 1] == myColumnChar)
                     ? MATCH
                     : MISMATCH);  // similarityScore(myLocString[i-1],myColumnChar);//seqB[j-1]
            traceback[1] = _curr_F;
            traceback[2] = _curr_E;
            traceback[3] = 0;

            _curr_H = findMax(traceback, 4, &ind);

            thread_max_i = (thread_max >= _curr_H) ? thread_max_i : i;
            thread_max_j = (thread_max >= _curr_H) ? thread_max_j : myTId + 1;
            thread_max   = (thread_max >= _curr_H) ? thread_max : _curr_H;

            i++;
        }

        __syncthreads();
    }
    __syncthreads();

    thread_max = blockShuffleReduce(thread_max, thread_max_i, thread_max_j,
                                    minSize);  // thread 0 will have the correct values

    __syncthreads();

    if(myTId == 0)
    {
        if(lengthSeqA < lengthSeqB)
        {
            seqB_align_begin[myId] = newlengthSeqB - (thread_max_i);
            seqA_align_begin[myId] = newlengthSeqA - (thread_max_j);
        }
        else
        {
            seqA_align_begin[myId] = newlengthSeqA - (thread_max_i);
            seqB_align_begin[myId] = newlengthSeqB - (thread_max_j);
        }
    }
    __syncthreads();
}
