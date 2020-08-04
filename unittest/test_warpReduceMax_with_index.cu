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
__global__ void test_warpReduceMax_with_index(
  short *values,
  short *indices1,
  short *indices2,
  const unsigned seq_length
){
  const int laneId = threadIdx.x % 32;
  const auto maxval = warpReduceMax_with_index<DIR>(values[laneId], indices1[laneId], indices2[laneId], seq_length);
  values[laneId] = maxval;
}



TEST_CASE("warpReduceMax_with_index forward general"){
  std::mt19937 gen;
  std::uniform_int_distribution<short> value_dist(std::numeric_limits<short>::min(),std::numeric_limits<short>::max());
  std::uniform_int_distribution<short> len_dist(1,32);

  std::vector<short> h_values(32);

  std::vector<short> h_index1(32);
  std::vector<short> h_index2(32);
  for(int i=0;i<32;i++){
    h_index1[i] = i;
    h_index2[i] = 32-i;
  }

  thrust::device_vector<short> d_values;
  thrust::device_vector<short> d_index1;
  thrust::device_vector<short> d_index2;

  for(int i=0;i<10000;i++){
    for(int i=0;i<32;i++){
      h_values[i] = value_dist(gen);
    }

    d_values = h_values;
    d_index1 = h_index1;
    d_index2 = h_index2;

    const int len = len_dist(gen);

    test_warpReduceMax_with_index<Direction::FORWARD><<<1,32>>>(thrust::raw_pointer_cast(d_values.data()), thrust::raw_pointer_cast(d_index1.data()), thrust::raw_pointer_cast(d_index2.data()), len);

    cudaDeviceSynchronize();

    const auto maxiter = std::max_element(h_values.begin(), h_values.begin()+len);
    CHECK(d_values[0]==*maxiter);
    CHECK(d_index1[0]==(maxiter-h_values.begin()));
    CHECK(d_index2[0]==32-(maxiter-h_values.begin()));
    CHECK(d_index1[0]<len);
  }
}



TEST_CASE("warpReduceMax_with_index forward duplicates"){
  std::mt19937 gen;
  std::uniform_int_distribution<short> value_dist(-3,3);
  std::uniform_int_distribution<short> len_dist(1,32);

  std::vector<short> h_values(32);

  std::vector<short> h_index1(32);
  std::vector<short> h_index2(32);
  for(int i=0;i<32;i++){
    h_index1[i] = i;
    h_index2[i] = 32-i;
  }

  thrust::device_vector<short> d_values;
  thrust::device_vector<short> d_index1;
  thrust::device_vector<short> d_index2;

  for(int i=0;i<1000;i++){
    for(int i=0;i<32;i++){
      h_values[i] = value_dist(gen);
    }

    d_values = h_values;
    d_index1 = h_index1;
    d_index2 = h_index2;

    const int len = len_dist(gen);

    test_warpReduceMax_with_index<Direction::FORWARD><<<1,32>>>(thrust::raw_pointer_cast(d_values.data()), thrust::raw_pointer_cast(d_index1.data()), thrust::raw_pointer_cast(d_index2.data()), len);

    cudaDeviceSynchronize();

    //Note that max_element returns the first index if the same maximum value is found multiple times
    const auto maxiter = std::max_element(h_values.begin(), h_values.begin()+len);
    CHECK(d_values[0]==*maxiter);
    CHECK(d_index1[0]==(maxiter-h_values.begin()));
    CHECK(d_index2[0]==32-(maxiter-h_values.begin()));
    CHECK(d_index1[0]<len);
  }
}



TEST_CASE("warpReduceMax_with_index reverse general"){
  std::mt19937 gen;
  std::uniform_int_distribution<short> value_dist(std::numeric_limits<short>::min(),std::numeric_limits<short>::max());
  std::uniform_int_distribution<short> len_dist(1,32);

  std::vector<short> h_values(32);

  std::vector<short> h_index1(32);
  std::vector<short> h_index2(32);
  for(int i=0;i<32;i++){
    h_index1[i] = i;
    h_index2[i] = 32-i;
  }

  thrust::device_vector<short> d_values;
  thrust::device_vector<short> d_index1;
  thrust::device_vector<short> d_index2;

  for(int i=0;i<10000;i++){
    for(int i=0;i<32;i++){
      h_values[i] = value_dist(gen);
    }

    d_values = h_values;
    d_index1 = h_index1;
    d_index2 = h_index2;

    const int len = len_dist(gen);

    test_warpReduceMax_with_index<Direction::REVERSE><<<1,32>>>(thrust::raw_pointer_cast(d_values.data()), thrust::raw_pointer_cast(d_index1.data()), thrust::raw_pointer_cast(d_index2.data()), len);

    cudaDeviceSynchronize();

    const auto maxiter = std::max_element(h_values.begin(), h_values.begin()+len);
    CHECK(d_values[0]==*maxiter);
    CHECK(d_index1[0]==maxiter-h_values.begin());
    CHECK(d_index2[0]==32-(maxiter-h_values.begin()));
    CHECK(d_index1[0]<len);
  }
}



TEST_CASE("warpReduceMax_with_index reverse duplicates"){
  std::mt19937 gen;
  std::uniform_int_distribution<short> value_dist(-3,3);
  std::uniform_int_distribution<short> len_dist(1,32);

  std::vector<short> h_values(32);

  std::vector<short> h_index1(32);
  std::vector<short> h_index2(32);
  for(int i=0;i<32;i++){
    h_index1[i] = i;
    h_index2[i] = 32-i;
  }

  thrust::device_vector<short> d_values;
  thrust::device_vector<short> d_index1;
  thrust::device_vector<short> d_index2;

  for(int i=0;i<1000;i++){
    for(int i=0;i<32;i++){
      h_values[i] = value_dist(gen);
    }

    d_values = h_values;
    d_index1 = h_index1;
    d_index2 = h_index2;

    const int len = len_dist(gen);

    test_warpReduceMax_with_index<Direction::REVERSE><<<1,32>>>(thrust::raw_pointer_cast(d_values.data()), thrust::raw_pointer_cast(d_index1.data()), thrust::raw_pointer_cast(d_index2.data()), len);

    cudaDeviceSynchronize();

    //Note that max_element returns the first index if the same maximum value is found multiple times
    const auto maxiter = std::max_element(h_values.begin(), h_values.begin()+len);
    CHECK(d_values[0]==*maxiter);
    CHECK(d_index1[0]==maxiter-h_values.begin());
    CHECK(d_index2[0]==32-(maxiter-h_values.begin()));
    CHECK(d_index1[0]<len);
  }
}
