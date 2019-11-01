
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
        // val     = max(val, __shfl_down_sync(mask, val, offset));
        // newInd  = __shfl_down_sync(mask, ind, offset);
        // newInd2 = __shfl_down_sync(mask, ind2, offset);
        int tempVal = __shfl_down_sync(mask, val, offset);
        val     = max(val,tempVal);
        newInd  = __shfl_down_sync(mask, ind, offset);
        newInd2 = __shfl_down_sync(mask, ind2, offset);
      //  __syncthreads();

        // if(val != myMax)
        // {
        //     ind   = newInd;
        //     ind2  = newInd2;
        //     myMax = val;
        // }
        if(val != myMax)
        {
            ind   = newInd;
            ind2  = newInd2;
            myMax = val;
        }else if((val == tempVal) )
        {
        //  printf("elif, val:%d, ind:%d, ind2:%d \n", val, ind, ind2);
          if(newInd < ind){
            ind = newInd;
            ind2 = newInd2;
          }
        }
      //  __syncthreads();
    }
//    __syncthreads();
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
    unsigned maxSize = lengthSeqA > lengthSeqB ? lengthSeqA : lengthSeqB;

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
if(lengthSeqA < lengthSeqB){
  seqB_align_begin[myId] = current_i;
  seqA_align_begin[myId] = current_j;
}else{
  seqA_align_begin[myId] = current_i;
  seqB_align_begin[myId] = current_j;
}

    // printf("traceback done\n");

    // *tick_out = tick;
}

__global__ void
align_sequences_gpu(char* seqA_array, char* seqB_array, unsigned* prefix_lengthA,
                    unsigned* prefix_lengthB,  short* seqA_align_begin, short* seqA_align_end,
                    short* seqB_align_begin, short* seqB_align_end)
{
    int myId  = blockIdx.x;
    int myTId = threadIdx.x;
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

     // short* curr_H =
     //     (short*) (&is_valid_array[3 * minSize +
     //                               (minSize & 1)]);  // point where the valid_array ends
    // short* prev_H      = &curr_H[minSize + 1];      // where the curr_H array ends
    // short* prev_prev_H = &prev_H[minSize + 1];
    // totBytes += (3 * minSize + (minSize & 1)) * sizeof(char) +
    //             ((minSize + 1) + (minSize + 1)) * sizeof(short);
    //
    // short* curr_E      = &prev_prev_H[minSize + 1];
    // short* prev_E      = &curr_E[minSize + 1];
    // short* prev_prev_E = &prev_E[minSize + 1];
    // totBytes += ((minSize + 1) + (minSize + 1) + (minSize + 1)) * sizeof(short);
    //
    // short* curr_F      = &prev_prev_E[minSize + 1];
    // short* prev_F      = &curr_F[minSize + 1];
    // short* prev_prev_F = &prev_F[minSize + 1];
    // totBytes += ((minSize + 1) + (minSize + 1) + (minSize + 1)) * sizeof(short);
    // short* curr_H =
    //     (short*) (&is_valid_array[3 * minSize +
    //                               (minSize & 1)]);
    char* myLocString = (char*) &is_valid[3 * minSize + (minSize & 1)];
  //  totBytes += (minSize + 1) * sizeof(short) + (maxSize) * sizeof(char);

    //unsigned      alignmentPad = 4 + (4 - totBytes % 4);
    // unsigned int* diagOffset   = (unsigned int*) &myLocString[maxSize + alignmentPad];
    // char* v = is_valid;


    __syncthreads();
    memset(is_valid, 0, minSize);
    is_valid += minSize;
    memset(is_valid, 1, minSize);
    is_valid += minSize;
    memset(is_valid, 0, minSize);

  //  memset(curr_H, 0, 9 * (minSize + 1) * sizeof(short));

    // __shared__ int i_max;
    // __shared__ int j_max;
    int            j            = myTId + 1;
      char myColumnChar;
    if(lengthSeqA < lengthSeqB){
       myColumnChar = seqA[j - 1];  // read only once
      for(int i = myTId; i < lengthSeqB; i += 32)
      {
          myLocString[i] = seqB[i];
      }
      }
    else{
       myColumnChar = seqB[j - 1];
      for(int i = myTId; i < lengthSeqA; i += 32)
      {
          myLocString[i] = seqA[i];
      }
      }


    __syncthreads();

    short            traceback[4];
    // __shared__ short iVal[4];  //= {-1,-1,0,0};
    // iVal[0] = -1;
    // iVal[1] = -1;
    // iVal[2] = 0;
    // iVal[3] = 0;
    // __shared__ short jVal[4];  //= {-1,0,-1,0};
    // jVal[0] = -1;
    // jVal[1] = 0;
    // jVal[2] = -1;
    // jVal[3] = 0;

    int ind;

    int   i            = 1;
    short thread_max   = 0;
    short thread_max_i = 0;
    short thread_max_j = 0;

  //  short* tmp_ptr;
    short _curr_H = 0, _curr_F = 0, _curr_E = 0;
    short _prev_H = 0, _prev_F = 0, _prev_E = 0;
    short _prev_prev_H = 0, _prev_prev_F = 0, _prev_prev_E = 0;
    short _temp_Val = 0;
   __shared__ short sh_prev_E[12];
   __shared__ short sh_prev_H[12];
   __shared__ short sh_prev_prev_H[12];
  __shared__ short sh_last_prev_E , sh_last_prev_H, sh_last_prev_prev_H;
sh_prev_E[0]=0;
sh_prev_E[1]=0;
sh_prev_H[0]=0;
sh_prev_H[1]=0;
sh_prev_prev_H[0]=0;
sh_prev_prev_H[1] = 0;

   //for(int j = 0; j < 2; j++) printf("valE:%d, valH:%d, valprevH:%d\n",sh_prev_E[j], sh_prev_H[j], sh_prev_prev_H[j]);

    __syncthreads();
    for(int diag = 0; diag < lengthSeqA + lengthSeqB - 1; diag++)
    {  // iterate for the number of anti-diagonals
//__syncthreads();
        is_valid = is_valid - (diag < minSize || diag >= maxSize);

       // tmp_ptr     = prev_H;
       // prev_H      = curr_H;
       // curr_H      = prev_prev_H;
       // prev_prev_H = tmp_ptr;
	_temp_Val = _prev_H;
	_prev_H = _curr_H;
	_curr_H = _prev_prev_H;
	_prev_prev_H = _temp_Val;

	_curr_H = 0;

//        memset(curr_H, 0, (minSize + 1) * sizeof(short));
        __syncthreads();
//        tmp_ptr     = prev_E;
 //       prev_E      = curr_E;
  //      curr_E      = prev_prev_E;
   //     prev_prev_E = tmp_ptr;
        _temp_Val = _prev_E;
        _prev_E = _curr_E;
        _curr_E = _prev_prev_E;
        _prev_prev_E = _temp_Val;
        //memset(curr_E, 0, (minSize + 1) * sizeof(short));
	_curr_E = 0;
 __syncthreads();
      //  tmp_ptr     = prev_F;
       // prev_F      = curr_F;
       // curr_F      = prev_prev_F;
       // prev_prev_F = tmp_ptr;
        _temp_Val = _prev_F;
        _prev_F = _curr_F;
        _curr_F = _prev_prev_F;
        _prev_prev_F = _temp_Val;
       // memset(curr_F, 0, (minSize + 1) * sizeof(short));
	_curr_F = 0;
            short laneId = threadIdx.x%32;
            short warpId = threadIdx.x/32;
           if(laneId == 31){
        //        if(threadIdx.x == 31 || threadIdx.x == 32) printf("threadId:%d, prev_e:%d, prev_H:%d, diag:%d\n",threadIdx.x, _prev_E,_prev_H, diag);
                sh_prev_E[warpId] = _prev_E;
                sh_prev_H[warpId] = _prev_H;
		sh_prev_prev_H[warpId] = _prev_prev_H;
        }

         if(threadIdx.x == blockDim.x -2){
		sh_last_prev_E = _prev_E;
		sh_last_prev_H = _prev_H;
		sh_last_prev_prev_H = _prev_prev_H;
	}
          __syncthreads();

  // if((threadIdx.x == 31 || threadIdx.x == 32) && (diag == 32 || diag == 33)) printf("threadId:%d, sh_prev_E[0]:%d, she_prev_E[1]:%d diag:%d\n", threadIdx.x, sh_prev_E[0], sh_prev_E[1], diag);

        if(is_valid[myTId] && myTId < minSize)
        {

       /*   if(laneId == 31){
               // printf("warpId:%d, prev_e:%d,prev_h:%d, prev_prev_h:%d\n",warpId, _prev_E,_prev_H,_prev_prev_H);
                sh_prev_E[warpId] = _prev_E;
                sh_prev_H[warpId] = _prev_H;
                sh_prev_prev_H[warpId] = _prev_prev_H;
        } */
// short fVal  = prev_F[j] + EXTEND_GAP;
           // short hfVal = prev_H[j] + START_GAP;
           // short eVal  = prev_E[j - 1] + EXTEND_GAP;
           // short heVal = prev_H[j - 1] + START_GAP;
      //     printf("kernel here from thread:%d\n", threadIdx.x);
	    unsigned mask  = __ballot_sync(__activemask(), (is_valid[myTId] &&( myTId < minSize)));

//printf("mask:%x diag:%d\n",mask,diag);
	    short fVal = _prev_F + EXTEND_GAP;
	    short hfVal = _prev_H + START_GAP;
	    //short laneId = threadIdx.x%32;
	    //short warpId = threadIdx.x/32;
	    short valeShfl = __shfl_sync(mask, _prev_E, laneId- 1, 32);
	    short valheShfl = __shfl_sync(mask, _prev_H, laneId - 1, 32);


            short eVal =((warpId !=0 && laneId == 0)?sh_prev_E[warpId-1]: valeShfl) + EXTEND_GAP;
	    short heVal =((warpId !=0 && laneId == 0)?sh_prev_H[warpId-1]:valheShfl) + START_GAP;
            if(warpId == 0 && laneId == 0){
              eVal = 0 + EXTEND_GAP;
              heVal = 0 + START_GAP;
            }
	         if(threadIdx.x == blockDim.x -1){
                eVal = sh_last_prev_E+EXTEND_GAP;
                heVal = sh_last_prev_H + START_GAP;
               // sh_last_prev_prev_H = _prev_prev_H;
        }
          //if(threadIdx.x == 0)printf("eVal:%d, heVal:%d, diag:%d,i:%d, j:%d\n",eVal, heVal, diag, i, j);
	//__syncthreads();
 	//curr_F[j] = (fVal > hfVal) ? fVal : hfVal;
		_curr_F = (fVal > hfVal) ? fVal : hfVal;
		_curr_E = (eVal > heVal) ? eVal : heVal;
 //curr_E[j] = (eVal > heVal) ? eVal : heVal;

short testShufll = __shfl_sync(mask, _prev_prev_H, laneId - 1, 32);
	   short valPPh = (warpId !=0 && laneId == 0)?sh_prev_prev_H[warpId-1]:testShufll;
   if(warpId == 0 && laneId == 0) valPPh = 0;
                 if(threadIdx.x == blockDim.x -1){
                valPPh = sh_last_prev_prev_H;// = _prev_prev_H;
        }
//if(threadIdx.x == 32&& i == 40)printf ("from 32 i = 1 prev_H val:%d\n",sh_prev_prev_H[0]);
//if(threadIdx.x==31 && i==41)printf("from thrd 31 i = 2 prev_H:%d\n",_prev_prev_H);

//if(threadIdx.x == 31)printf("thread:%d, valPPH:%d,eval:%d, heVal:%d, sh_prev_E:%d, sh_prev_H:%d, wapr:%d,diag:%d\n",threadIdx.x, valPPh,eVal,heVal, sh_prev_E[warpId-1], sh_prev_H[warpId-1],warpId,diag);

//__syncthreads();

     // if(threadIdx.x == 32) printf("sheVal[1]:%d, sheVal[2]:%d\n",sh_prev_E[0], sh_prev_E[1]);
            traceback[0] =
                valPPh +
                ((myLocString[i - 1] == myColumnChar)
                     ? MATCH
                     : MISMATCH);  // similarityScore(myLocString[i-1],myColumnChar);//seqB[j-1]
            traceback[1] = _curr_F;
            traceback[2] = _curr_E;
            traceback[3] = 0;

            _curr_H = findMax(traceback, 4, &ind);
//	if((i==33 || i == 32 || i == 31) && (j == 33 || j == 31 || j ==32) )
//		printf("i:%d, j:%d, myColChar:%c, myRowChar:%c, _curr_H:%d, valPPH:%d, _curr_F:%d, _curr_E:%d, testShuffl:%d, prev_prev_H:%d, sh_prev_prevH:%d\n",i,j,myLocString[i-1], myColumnChar, _curr_H, valPPh, _curr_F, _curr_E,testShufll,_prev_prev_H,sh_prev_prev_H[0]);

 //
		//if(warpId != 0 && laneId == 0)printf("diag:%d, _curr_e:%d, curr_f:%d, curr_h:%d\n", diag, _curr_E, _curr_F, _curr_H);


            thread_max_i = (thread_max >= _curr_H) ? thread_max_i : i;
            thread_max_j = (thread_max >= _curr_H) ? thread_max_j : myTId + 1;
            thread_max   = (thread_max >= _curr_H) ? thread_max : _curr_H;
        //    printf("thread:%d, max:%d\n",myTId, thread_max);
//	 if(is_valid[myTId] && myTId < minSize)
            i++;
        }

__syncthreads();
//if(blockIdx.x == 0)printf("max:%d, thread_i:%d, thread_j: %d\n", thread_max, thread_max_i, thread_max_j);
        //__syncthreads();
    }
    __syncthreads();

    thread_max = blockShuffleReduce(thread_max, thread_max_i, thread_max_j,
                                    minSize);  // thread 0 will have the correct values

    __syncthreads();



    if(myTId == 0)
    {


        if(lengthSeqA < lengthSeqB){
          seqB_align_end[myId] = thread_max_i;
          seqA_align_end[myId] = thread_max_j;
        }else{
        seqA_align_end[myId] = thread_max_i;
        seqB_align_end[myId] = thread_max_j;
        }

}

__syncthreads();
//if(threadIdx.x == 0) printf("Aend: %d, Bend: %d\n",seqA_align_end[0], seqB_align_end[0]);

j            = myTId + 1;

 int newlengthSeqA = seqA_align_end[myId];
 int newlengthSeqB = seqB_align_end[myId];
int seqAend = newlengthSeqA -1;
int seqBend = newlengthSeqB -1;
__syncthreads();


if(lengthSeqA < lengthSeqB){
  if(myTId < newlengthSeqA)
    seqA[seqAend - (j-1)] = myColumnChar;

  for(int i = myTId; i < newlengthSeqB; i+=32){
    seqB[seqBend - i] = myLocString[i];
  }

}else{
  if(myTId < newlengthSeqB)
  seqB[seqBend - (j-1)] = myColumnChar;


  for(int i = myTId; i < newlengthSeqA; i+=32){
    seqA[seqAend - i] = myLocString[i];
  }
}

__syncthreads();


maxSize = newlengthSeqA > newlengthSeqB ? newlengthSeqA : newlengthSeqB;
minSize = newlengthSeqA < newlengthSeqB ? newlengthSeqA : newlengthSeqB;
__syncthreads();
     // curr_H =
     //    (short*) (&is_valid_array[3 * minSize +
     //                              (minSize & 1)]);  // point where the valid_array ends
     // prev_H      = &curr_H[minSize + 1];      // where the curr_H array ends
     // prev_prev_H = &prev_H[minSize + 1];
     //
     //
     // curr_E      = &prev_prev_H[minSize + 1];
     // prev_E      = &curr_E[minSize + 1];
     // prev_prev_E = &prev_E[minSize + 1];
     //
     // curr_F      = &prev_prev_E[minSize + 1];
     // prev_F      = &curr_F[minSize + 1];
     // prev_prev_F = &prev_F[minSize + 1];
     //
     // myLocString = (char*) &prev_prev_F[minSize + 1];
__syncthreads();

//            memset(curr_H, 0, 9 * (minSize + 1) * sizeof(short));

           is_valid = &is_valid_array[0]; //reset is_valid array for second iter
           memset(is_valid, 0, minSize);
           is_valid += minSize;
           memset(is_valid, 1, minSize);
           is_valid += minSize;
           memset(is_valid, 0, minSize);
__syncthreads();

if(lengthSeqA < lengthSeqB){
   if(j < newlengthSeqA);
    myColumnChar = seqA[j - 1];  // read only once
  for(int i = myTId; i < newlengthSeqB; i += 32)
  {
      myLocString[i] = seqB[i]; // locString contains reference/longer string
  }
  }
else{
  if(j < newlengthSeqB);
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
    //   short _temp_Val = 0;
    //  __shared__ short sh_prev_E[2];
    //  __shared__ short sh_prev_H[2];
    //  __shared__ short sh_prev_prev_H[2];
    // __shared__ short sh_last_prev_E , sh_last_prev_H, sh_last_prev_prev_H;
  sh_prev_E[0]=0;
  sh_prev_E[1]=0;
  sh_prev_H[0]=0;
  sh_prev_H[1]=0;
  sh_prev_prev_H[0]=0;
  sh_prev_prev_H[1] = 0;

     //for(int j = 0; j < 2; j++) printf("valE:%d, valH:%d, valprevH:%d\n",sh_prev_E[j], sh_prev_H[j], sh_prev_prev_H[j]);

      __syncthreads();
      for(int diag = 0; diag <  newlengthSeqA + newlengthSeqB - 1; diag++)
      {  // iterate for the number of anti-diagonals
  //__syncthreads();
          is_valid = is_valid - (diag < minSize || diag >= maxSize);

         // tmp_ptr     = prev_H;
         // prev_H      = curr_H;
         // curr_H      = prev_prev_H;
         // prev_prev_H = tmp_ptr;
  	_temp_Val = _prev_H;
  	_prev_H = _curr_H;
  	_curr_H = _prev_prev_H;
  	_prev_prev_H = _temp_Val;

  	_curr_H = 0;

  //        memset(curr_H, 0, (minSize + 1) * sizeof(short));
          __syncthreads();
  //        tmp_ptr     = prev_E;
   //       prev_E      = curr_E;
    //      curr_E      = prev_prev_E;
     //     prev_prev_E = tmp_ptr;
          _temp_Val = _prev_E;
          _prev_E = _curr_E;
          _curr_E = _prev_prev_E;
          _prev_prev_E = _temp_Val;
          //memset(curr_E, 0, (minSize + 1) * sizeof(short));
  	_curr_E = 0;
   __syncthreads();
        //  tmp_ptr     = prev_F;
         // prev_F      = curr_F;
         // curr_F      = prev_prev_F;
         // prev_prev_F = tmp_ptr;
          _temp_Val = _prev_F;
          _prev_F = _curr_F;
          _curr_F = _prev_prev_F;
          _prev_prev_F = _temp_Val;
         // memset(curr_F, 0, (minSize + 1) * sizeof(short));
  	_curr_F = 0;
              short laneId = threadIdx.x%32;
              short warpId = threadIdx.x/32;
             if(laneId == 31){
          //        if(threadIdx.x == 31 || threadIdx.x == 32) printf("threadId:%d, prev_e:%d, prev_H:%d, diag:%d\n",threadIdx.x, _prev_E,_prev_H, diag);
                  sh_prev_E[warpId] = _prev_E;
                  sh_prev_H[warpId] = _prev_H;
  		sh_prev_prev_H[warpId] = _prev_prev_H;
          }

           if(threadIdx.x == blockDim.x -2){
  		sh_last_prev_E = _prev_E;
  		sh_last_prev_H = _prev_H;
  		sh_last_prev_prev_H = _prev_prev_H;
  	}
            __syncthreads();

    // if((threadIdx.x == 31 || threadIdx.x == 32) && (diag == 32 || diag == 33)) printf("threadId:%d, sh_prev_E[0]:%d, she_prev_E[1]:%d diag:%d\n", threadIdx.x, sh_prev_E[0], sh_prev_E[1], diag);

          if(is_valid[myTId] && myTId < minSize)
          {

         /*   if(laneId == 31){
                 // printf("warpId:%d, prev_e:%d,prev_h:%d, prev_prev_h:%d\n",warpId, _prev_E,_prev_H,_prev_prev_H);
                  sh_prev_E[warpId] = _prev_E;
                  sh_prev_H[warpId] = _prev_H;
                  sh_prev_prev_H[warpId] = _prev_prev_H;
          } */
  // short fVal  = prev_F[j] + EXTEND_GAP;
             // short hfVal = prev_H[j] + START_GAP;
             // short eVal  = prev_E[j - 1] + EXTEND_GAP;
             // short heVal = prev_H[j - 1] + START_GAP;
        //     printf("kernel here from thread:%d\n", threadIdx.x);
  	    unsigned mask  = __ballot_sync(__activemask(), (is_valid[myTId] &&( myTId < minSize)));

  //printf("mask:%x diag:%d\n",mask,diag);
  	    short fVal = _prev_F + EXTEND_GAP;
  	    short hfVal = _prev_H + START_GAP;
  	    //short laneId = threadIdx.x%32;
  	    //short warpId = threadIdx.x/32;
  	    short valeShfl = __shfl_sync(mask, _prev_E, laneId- 1, 32);
  	    short valheShfl = __shfl_sync(mask, _prev_H, laneId - 1, 32);


              short eVal =((warpId !=0 && laneId == 0)?sh_prev_E[warpId-1]: valeShfl) + EXTEND_GAP;
  	    short heVal =((warpId !=0 && laneId == 0)?sh_prev_H[warpId-1]:valheShfl) + START_GAP;
              if(warpId == 0 && laneId == 0){
                eVal = 0 + EXTEND_GAP;
                heVal = 0 + START_GAP;
              }
  	         if(threadIdx.x == blockDim.x -1){
                  eVal = sh_last_prev_E+EXTEND_GAP;
                  heVal = sh_last_prev_H + START_GAP;
                 // sh_last_prev_prev_H = _prev_prev_H;
          }
            //if(threadIdx.x == 0)printf("eVal:%d, heVal:%d, diag:%d,i:%d, j:%d\n",eVal, heVal, diag, i, j);
  	//__syncthreads();
   	//curr_F[j] = (fVal > hfVal) ? fVal : hfVal;
  		_curr_F = (fVal > hfVal) ? fVal : hfVal;
  		_curr_E = (eVal > heVal) ? eVal : heVal;
   //curr_E[j] = (eVal > heVal) ? eVal : heVal;

  short testShufll = __shfl_sync(mask, _prev_prev_H, laneId - 1, 32);
  	   short valPPh = (warpId !=0 && laneId == 0)?sh_prev_prev_H[warpId-1]:testShufll;
     if(warpId == 0 && laneId == 0) valPPh = 0;
                   if(threadIdx.x == blockDim.x -1){
                  valPPh = sh_last_prev_prev_H;// = _prev_prev_H;
          }
  //if(threadIdx.x == 32&& i == 40)printf ("from 32 i = 1 prev_H val:%d\n",sh_prev_prev_H[0]);
  //if(threadIdx.x==31 && i==41)printf("from thrd 31 i = 2 prev_H:%d\n",_prev_prev_H);

  //if(threadIdx.x == 31)printf("thread:%d, valPPH:%d,eval:%d, heVal:%d, sh_prev_E:%d, sh_prev_H:%d, wapr:%d,diag:%d\n",threadIdx.x, valPPh,eVal,heVal, sh_prev_E[warpId-1], sh_prev_H[warpId-1],warpId,diag);

  //__syncthreads();

       // if(threadIdx.x == 32) printf("sheVal[1]:%d, sheVal[2]:%d\n",sh_prev_E[0], sh_prev_E[1]);
              traceback[0] =
                  valPPh +
                  ((myLocString[i - 1] == myColumnChar)
                       ? MATCH
                       : MISMATCH);  // similarityScore(myLocString[i-1],myColumnChar);//seqB[j-1]
              traceback[1] = _curr_F;
              traceback[2] = _curr_E;
              traceback[3] = 0;

              _curr_H = findMax(traceback, 4, &ind);
  //	if((i==33 || i == 32 || i == 31) && (j == 33 || j == 31 || j ==32) )
  //		printf("i:%d, j:%d, myColChar:%c, myRowChar:%c, _curr_H:%d, valPPH:%d, _curr_F:%d, _curr_E:%d, testShuffl:%d, prev_prev_H:%d, sh_prev_prevH:%d\n",i,j,myLocString[i-1], myColumnChar, _curr_H, valPPh, _curr_F, _curr_E,testShufll,_prev_prev_H,sh_prev_prev_H[0]);

   //
  		//if(warpId != 0 && laneId == 0)printf("diag:%d, _curr_e:%d, curr_f:%d, curr_h:%d\n", diag, _curr_E, _curr_F, _curr_H);


              thread_max_i = (thread_max >= _curr_H) ? thread_max_i : i;
              thread_max_j = (thread_max >= _curr_H) ? thread_max_j : myTId + 1;
              thread_max   = (thread_max >= _curr_H) ? thread_max : _curr_H;
          //    printf("thread:%d, max:%d\n",myTId, thread_max);
  //	 if(is_valid[myTId] && myTId < minSize)
              i++;
          }

  __syncthreads();
  //if(blockIdx.x == 0)printf("max:%d, thread_i:%d, thread_j: %d\n", thread_max, thread_max_i, thread_max_j);
          //__syncthreads();
      }
      __syncthreads();

      thread_max = blockShuffleReduce(thread_max, thread_max_i, thread_max_j,
                                      minSize);  // thread 0 will have the correct values

      __syncthreads();


  // //    short* tmp_ptr;
  // //  locSum = 0;
  //
  // //    __syncthreads();
  //     for(int diag = 0; diag < newlengthSeqA + newlengthSeqB - 1; diag++)
  //     {  // iterate for the number of anti-diagonals
  //
  //         is_valid = is_valid - (diag < minSize || diag >= maxSize);
  //
  //         tmp_ptr     = prev_H;
  //         prev_H      = curr_H;
  //         curr_H      = prev_prev_H;
  //         prev_prev_H = tmp_ptr;
  //
  //         memset(curr_H, 0, (minSize + 1) * sizeof(short));
  //         __syncthreads();
  //         tmp_ptr     = prev_E;
  //         prev_E      = curr_E;
  //         curr_E      = prev_prev_E;
  //         prev_prev_E = tmp_ptr;
  //
  //         memset(curr_E, 0, (minSize + 1) * sizeof(short));
  //         __syncthreads();
  //         tmp_ptr     = prev_F;
  //         prev_F      = curr_F;
  //         curr_F      = prev_prev_F;
  //         prev_prev_F = tmp_ptr;
  //
  //         memset(curr_F, 0, (minSize + 1) * sizeof(short));
  //
  //         __syncthreads();
  //
  //         if(is_valid[myTId] && myTId < minSize)
  //         {
  //             short fVal  = prev_F[j] + EXTEND_GAP;
  //             short hfVal = prev_H[j] + START_GAP;
  //             short eVal  = prev_E[j - 1] + EXTEND_GAP;
  //             short heVal = prev_H[j - 1] + START_GAP;
  //
  //             curr_F[j] = (fVal > hfVal) ? fVal : hfVal;
  //             curr_E[j] = (eVal > heVal) ? eVal : heVal;
  //
  //         //    (myLocString[i-1] == myColumnChar)?MATCH:MISMATCH
  //
  //             traceback[0] =
  //                 prev_prev_H[j - 1] +
  //                 ((myLocString[i - 1] == myColumnChar)
  //                      ? MATCH
  //                      : MISMATCH);  // similarityScore(myLocString[i-1],myColumnChar);//seqB[j-1]
  //             traceback[1] = curr_F[j];
  //             traceback[2] = curr_E[j];
  //             traceback[3] = 0;
  //
  //             curr_H[j] = findMax(traceback, 4, &ind);
  //
  //             thread_max_i = (thread_max >= curr_H[j]) ? thread_max_i : i;
  //             thread_max_j = (thread_max >= curr_H[j]) ? thread_max_j : myTId + 1;
  //             thread_max   = (thread_max >= curr_H[j]) ? thread_max : curr_H[j];
  //
  //             i++;
  //         }
  //         __syncthreads();
  //     }
  //     __syncthreads();
  //
  //     thread_max = blockShuffleReduce(thread_max, thread_max_i, thread_max_j,
  //                                     minSize);  // thread 0 will have the correct values

  //    __syncthreads();


          if(myTId == 0)
          {
              if(lengthSeqA < lengthSeqB){
                seqB_align_begin[myId] = newlengthSeqB - (thread_max_i);
                seqA_align_begin[myId] = newlengthSeqA - (thread_max_j);
              }else{
              seqA_align_begin[myId] = newlengthSeqA - (thread_max_i);
              seqB_align_begin[myId] = newlengthSeqB - (thread_max_j);
              }
          }
           __syncthreads();
        //   if(threadIdx.x == 0) printf("Abegin: %d, Bbegin: %d\n",seqA_align_begin[0], seqB_align_begin[0]);


}
