#ifndef UTILS_HPP
#define UTILS_HPP

#include <omp.h>
#include <string>
#include <vector>

#define cudaErrchk(ans)                                                                  \
    {                                                                                    \
        gpuAssert((ans), __FILE__, __LINE__);                                            \
    }
inline void

gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if(code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d cpu:%d\n", cudaGetErrorString(code), file, line,omp_get_thread_num());
        if(abort)
            exit(code);
    }
}

//Allocate `count` items in page-locked memory
template<class T>
T* PageLockedMalloc(const size_t count){
    T *temp;
    cudaErrchk(cudaMallocHost(&temp, count*sizeof(T)));
    return temp;
}

//Allocate `count` items on device memory
template<class T>
T* DeviceMalloc(const size_t count){
    T *temp;
    cudaErrchk(cudaMalloc(&temp, count*sizeof(T)));
    return temp;
}

//Allocate `count` items on device memory
template<class T>
T* DeviceMalloc(const size_t count, const T *const host_data){
    T *temp;
    cudaErrchk(cudaMalloc(&temp, count*sizeof(T)));
    cudaErrchk(cudaMemcpy(temp, host_data, count*sizeof(T), cudaMemcpyHostToDevice));
    return temp;
}

size_t getMaxLength(const std::vector<std::string> &v);
int get_new_min_length(short* alAend, short* alBend, int blocksLaunched);
#endif