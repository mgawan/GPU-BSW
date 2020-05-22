#ifndef UTILS_HPP
#define UTILS_HPP
#include "driver.hpp"
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

unsigned getMaxLength (std::vector<std::string> v);
void initialize_alignments(gpu_bsw_driver::alignment_results *alignments, int max_alignments);
void free_alignments(gpu_bsw_driver::alignment_results *alignments);
#endif