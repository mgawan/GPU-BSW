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

size_t getMaxLength(const std::vector<std::string> &v);
int get_new_min_length(short* alAend, short* alBend, int blocksLaunched);
#endif