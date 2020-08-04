#include "doctest.h"

#include <gpu_bsw/kernel.hpp>

#include <thrust/device_vector.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <random>

using namespace gpu_bsw;

template<class T>
std::ostream& operator<<(std::ostream &out, thrust::device_vector<T> &vec){
  for(size_t i=0;i<vec.size();i++)
    out<<i<<" "<<vec[i]<<"\n";
  return out;
}

template<Direction DIR>
__global__ void test_blockShuffleReduce_with_index(
  short *values,
  short *indices1,
  short *indices2,
  const unsigned seq_length
){
  const auto maxval = blockShuffleReduce_with_index<DIR>(values[threadIdx.x], indices1[threadIdx.x], indices2[threadIdx.x], seq_length);
  values[threadIdx.x] = maxval;
}



TEST_CASE("blockShuffleReduce_with_index forward general"){
  std::mt19937 gen;
  std::uniform_int_distribution<short> value_dist(std::numeric_limits<short>::min(),std::numeric_limits<short>::max());
  std::uniform_int_distribution<short> len_dist(1,32);

  std::vector<short> h_values(1024);

  std::vector<short> h_index1(1024);
  std::vector<short> h_index2(1024);
  for(int i=0;i<1024;i++){
    h_index1[i] = i;
    h_index2[i] = 1024-i;
  }

  thrust::device_vector<short> d_values;
  thrust::device_vector<short> d_index1;
  thrust::device_vector<short> d_index2;

  for(int i=0;i<10000;i++){
    for(int i=0;i<1024;i++){
      h_values[i] = value_dist(gen);
    }

    d_values = h_values;
    d_index1 = h_index1;
    d_index2 = h_index2;

    const int len = len_dist(gen);

    test_blockShuffleReduce_with_index<Direction::FORWARD><<<1,1024>>>(thrust::raw_pointer_cast(d_values.data()), thrust::raw_pointer_cast(d_index1.data()), thrust::raw_pointer_cast(d_index2.data()), len);

    cudaDeviceSynchronize();

    const auto maxiter = std::max_element(h_values.begin(), h_values.begin()+len);
    CHECK(d_values[0]==*maxiter);
    CHECK(d_index1[0]==(maxiter-h_values.begin()));
    CHECK(d_index2[0]==1024-(maxiter-h_values.begin()));
    CHECK(d_index1[0]<len);
  }
}



TEST_CASE("blockShuffleReduce_with_index forward general"){
  std::mt19937 gen;
  std::uniform_int_distribution<short> value_dist(-10,10);
  std::uniform_int_distribution<short> len_dist(1,32);

  std::vector<short> h_values(1024);

  std::vector<short> h_index1(1024);
  std::vector<short> h_index2(1024);
  for(int i=0;i<1024;i++){
    h_index1[i] = i;
    h_index2[i] = 1024-i;
  }

  thrust::device_vector<short> d_values;
  thrust::device_vector<short> d_index1;
  thrust::device_vector<short> d_index2;

  for(int i=0;i<10000;i++){
    for(int i=0;i<1024;i++){
      h_values[i] = value_dist(gen);
    }

    d_values = h_values;
    d_index1 = h_index1;
    d_index2 = h_index2;

    const int len = len_dist(gen);

    test_blockShuffleReduce_with_index<Direction::FORWARD><<<1,1024>>>(thrust::raw_pointer_cast(d_values.data()), thrust::raw_pointer_cast(d_index1.data()), thrust::raw_pointer_cast(d_index2.data()), len);

    cudaDeviceSynchronize();

    const auto maxiter = std::max_element(h_values.begin(), h_values.begin()+len);
    CHECK(d_values[0]==*maxiter);
    CHECK(d_index1[0]==(maxiter-h_values.begin()));
    CHECK(d_index2[0]==1024-(maxiter-h_values.begin()));
    CHECK(d_index1[0]<len);
  }
}



TEST_CASE("blockShuffleReduce_with_index reverse general"){
  std::mt19937 gen;
  std::uniform_int_distribution<short> value_dist(std::numeric_limits<short>::min(),std::numeric_limits<short>::max());
  std::uniform_int_distribution<short> len_dist(1,32);

  std::vector<short> h_values(1024);

  std::vector<short> h_index1(1024);
  std::vector<short> h_index2(1024);
  for(int i=0;i<1024;i++){
    h_index1[i] = i;
    h_index2[i] = 1024-i;
  }

  thrust::device_vector<short> d_values;
  thrust::device_vector<short> d_index1;
  thrust::device_vector<short> d_index2;

  for(int i=0;i<10000;i++){
    for(int i=0;i<1024;i++){
      h_values[i] = value_dist(gen);
    }

    d_values = h_values;
    d_index1 = h_index1;
    d_index2 = h_index2;

    const int len = len_dist(gen);

    test_blockShuffleReduce_with_index<Direction::REVERSE><<<1,1024>>>(thrust::raw_pointer_cast(d_values.data()), thrust::raw_pointer_cast(d_index1.data()), thrust::raw_pointer_cast(d_index2.data()), len);

    cudaDeviceSynchronize();

    const auto maxiter = std::max_element(h_values.begin(), h_values.begin()+len);
    CHECK(d_values[0]==*maxiter);
    CHECK(d_index1[0]==(maxiter-h_values.begin()));
    CHECK(d_index2[0]==1024-(maxiter-h_values.begin()));
    CHECK(d_index1[0]<len);
  }
}



TEST_CASE("blockShuffleReduce_with_index reverse duplicates"){
  std::mt19937 gen;
  std::uniform_int_distribution<short> value_dist(-10,10);
  std::uniform_int_distribution<short> len_dist(1,32);

  std::vector<short> h_values(1024);

  std::vector<short> h_index1(1024);
  std::vector<short> h_index2(1024);
  for(int i=0;i<1024;i++){
    h_index1[i] = i;
    h_index2[i] = 1024-i;
  }

  thrust::device_vector<short> d_values;
  thrust::device_vector<short> d_index1;
  thrust::device_vector<short> d_index2;

  for(int i=0;i<10000;i++){
    for(int i=0;i<1024;i++){
      h_values[i] = value_dist(gen);
    }

    d_values = h_values;
    d_index1 = h_index1;
    d_index2 = h_index2;

    const int len = len_dist(gen);

    test_blockShuffleReduce_with_index<Direction::REVERSE><<<1,1024>>>(thrust::raw_pointer_cast(d_values.data()), thrust::raw_pointer_cast(d_index1.data()), thrust::raw_pointer_cast(d_index2.data()), len);

    cudaDeviceSynchronize();

    const auto maxiter = std::max_element(h_values.begin(), h_values.begin()+len);
    CHECK(d_values[0]==*maxiter);
    CHECK(d_index1[0]==(maxiter-h_values.begin()));
    CHECK(d_index2[0]==1024-(maxiter-h_values.begin()));
    CHECK(d_index1[0]<len);
  }
}