#include <gpu_bsw/kernel.hpp>
#include <gpu_bsw/utils.hpp>

#include <thrust/device_vector.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace gpu_bsw;

std::mt19937 gen;
std::uniform_int_distribution<short> dna_dist(0,3);
std::uniform_int_distribution<short> len_dist(1,32);

std::vector<char> randomDNA(const int len){
  constexpr std::array<char, 4> bases = {'A','C','G','T'};
  std::vector<char> temp;
  temp.reserve(len);
  for(int i=0;i<len;i++){
    temp.push_back(bases.at(dna_dist(gen)));
  }
  return temp;
}



template<class T>
std::ostream& operator<<(std::ostream &out, thrust::device_vector<T> &vec){
  for(size_t i=0;i<vec.size();i++)
    out<<vec[i]<<" ";
  return out;
}



int SimpleSmithWaterman(
  const std::string &seqa,
  const std::string &seqb,
  const int gap_open,
  const int gap_extend,
  const int match_score,
  const int mismatch_score
){
  const auto size = seqa.size()*seqb.size();
  std::vector<int> E(size,0);
  std::vector<int> F(size,0);
  std::vector<int> H(size,0);
  const auto width = seqb.size();

  const auto c2d = [&](const size_t i, const size_t j){ return i*width+j; };
  const auto mm = [&](const char a, const char b){ return (a==b)?match_score:mismatch_score; };

  for(size_t i=0;i<seqa.size();i++)
  for(size_t j=0;j<seqb.size();j++){
    E[c2d(i,j)] = std::max(
      (j>0) ? E[c2d(i,j-1)] + gap_extend : 0,
      (j>0) ? H[c2d(i,j-1)] + gap_open   : 0
    );

    F[c2d(i,j)] = std::max(
      (i>0) ? F[c2d(i-1,j)] + gap_extend : 0,
      (i>0) ? H[c2d(i-1,j)] + gap_open   : 0
    );

    H[c2d(i,j)] = std::max(
      std::max(0, E[c2d(i,j)]),
      std::max(
        -1000, //F[c2d(i,j)],
        (i>0 && j>0) ? H[c2d(i-1,j-1)]+mm(seqa[i], seqb[j]) : 0
      )
    );
  }

  std::cout<<"F:\n";
  for(size_t i=0;i<seqa.size();i++){
    for(size_t j=0;j<seqb.size();j++){
      if(F[c2d(i,j)]>H[c2d(i,j)]+gap_open)
        std::cout<<"\033[91m";
      std::cout<<std::setw(2)<<F[c2d(i,j)]<<"|"<<H[c2d(i,j)]<<" ";
      std::cout<<"\033[39m";
    }
    std::cout<<"\n";
  }

  return *std::max_element(H.begin(), H.end());
}



int SSWClone(
  const std::string &seqa,
  const std::string &seqb,
  const int gap_open,
  const int gap_extend,
  const int match_score,
  const int mismatch_score
){
  const auto size = seqa.size()*seqb.size();
  std::vector<int> E(size,0);
  std::vector<int> F(size,0);
  std::vector<int> H(size,0);
  const auto width = seqb.size();

  const auto c2d = [&](const size_t i, const size_t j){ return i*width+j; };
  const auto mm = [&](const char a, const char b){ return (a==b)?match_score:mismatch_score; };

  for(size_t i=0;i<seqa.size();i++)
  for(size_t j=0;j<seqb.size();j++){
    E[c2d(i,j)] = std::max(
      (j>0) ? E[c2d(i,j-1)] + gap_extend : 0,
      (j>0) ? H[c2d(i,j-1)] + gap_open   : 0
    );

    F[c2d(i,j)] = std::max(
      (i>0) ? F[c2d(i-1,j)] + gap_extend : 0,
      (i>0) ? H[c2d(i-1,j)] + gap_open   : 0
    );

    H[c2d(i,j)] = std::max(
      std::max(0, E[c2d(i,j)]),
      std::max(
        -1000, //F[c2d(i,j)],
        (i>0 && j>0) ? H[c2d(i-1,j-1)]+mm(seqa[i], seqb[j]) : 0
      )
    );
  }

  std::cout<<"F:\n";
  for(size_t i=0;i<seqa.size();i++){
    for(size_t j=0;j<seqb.size();j++){
      if(F[c2d(i,j)]>H[c2d(i,j)]+gap_open)
        std::cout<<"\033[91m";
      std::cout<<std::setw(2)<<F[c2d(i,j)]<<"|"<<H[c2d(i,j)]<<" ";
      std::cout<<"\033[39m";
    }
    std::cout<<"\n";
  }

  return *std::max_element(H.begin(), H.end());
}



int main(){
  const int match_score    = 1;
  const int mismatch_score = -4;
  const int gap_open       = -6;
  const int gap_extend     = -1;

  const std::string seqa = "TATTATTAGCAGCAGTTTTTAGTTTACTTCAA";
  const std::string seqb = "CATGTTTGGTACTATGGCTGGCCCGAACGTACCTAAATTTGACTTCAGCACATATAACCCCCGTGTTTCGGAGGAGACCTTGCTGTAATTCTGTGCTCTGGAATAACTTTCCTCAATACTATC";

  std::cout<<SimpleSmithWaterman(seqa, seqb, gap_open, gap_extend, match_score, mismatch_score)<<std::endl;
}


// int main(){
//   const int alen = 40;
//   const int blen = 100;

//   thrust::device_vector<char> seqA_array;
//   seqA_array = randomDNA( alen);
//   thrust::device_vector<char> seqB_array;
//   seqB_array = randomDNA(blen);
//   thrust::device_vector<unsigned> prefix_lengthA(1);
//   thrust::device_vector<unsigned> prefix_lengthB(1);

//   prefix_lengthA[0] = alen;
//   prefix_lengthB[0] = blen;

//   thrust::device_vector<short> seqA_align_begin(1);
//   thrust::device_vector<short> seqA_align_end(1);
//   thrust::device_vector<short> seqB_align_begin(1);
//   thrust::device_vector<short> seqB_align_end(1);

//   thrust::device_vector<short> top_scores(1);

//   thrust::device_vector<short> scoring_matrix(2);
//   thrust::device_vector<short> encoding_matrix(1);

//   scoring_matrix[0] = 1;
//   scoring_matrix[1] = -4;

//   encoding_matrix[0] = 0;

//   const short startGap = 6;
//   const short extendGap = 1;

//   // unsigned minSize = (maxReadSize < maxContigSize) ? maxReadSize : maxContigSize;
//   const unsigned minSize = alen;
//   const unsigned totShmem = 3 * (minSize + 1) * sizeof(short);
//   const unsigned alignmentPad = 4 + (4 - totShmem % 4);
//   const size_t   ShmemBytes = totShmem + alignmentPad;
//   // if(ShmemBytes > 48000 && DT==DataType::DNA)
//       // cudaFuncSetAttribute(gpu_bsw::sequence_process<DataType::DNA,Direction::FORWARD>, cudaFuncAttributeMaxDynamicSharedMemorySize, ShmemBytes);

//   cudaErrchk(cudaDeviceSynchronize());

//   sequence_process<DataType::DNA, Direction::FORWARD><<<1, minSize, ShmemBytes>>>(
//     thrust::raw_pointer_cast(seqA_array.data()),
//     thrust::raw_pointer_cast(seqB_array.data()),
//     thrust::raw_pointer_cast(prefix_lengthA.data()),
//     thrust::raw_pointer_cast(prefix_lengthB.data()),
//     thrust::raw_pointer_cast(seqA_align_begin.data()),
//     thrust::raw_pointer_cast(seqA_align_end.data()),
//     thrust::raw_pointer_cast(seqB_align_begin.data()),
//     thrust::raw_pointer_cast(seqB_align_end.data()),
//     thrust::raw_pointer_cast(top_scores.data()),
//     startGap,
//     extendGap,
//     thrust::raw_pointer_cast(scoring_matrix.data()),
//     thrust::raw_pointer_cast(encoding_matrix.data())
//   );
//   cudaErrchk(cudaGetLastError());

//   cudaErrchk(cudaDeviceSynchronize());

//   std::cout<<"seqA             = "<<seqA_array<<"\n";
//   std::cout<<"seqB             = "<<seqB_array<<"\n";
//   std::cout<<"seqA_align_begin = "<<seqA_align_begin<<"\n";
//   std::cout<<"seqA_align_end   = "<<seqA_align_end<<"\n";
//   std::cout<<"seqB_align_begin = "<<seqB_align_begin<<"\n";
//   std::cout<<"seqB_align_end   = "<<seqB_align_end<<"\n";
//   std::cout<<"top_scores       = "<<top_scores<<"\n";
// }