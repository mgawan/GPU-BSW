#ifndef DRIVER_HPP
#define DRIVER_HPP

#include "kernel.hpp"
#include <chrono>
#include <cmath>
#include <string>
#include <vector>
#include <omp.h>

#define NOW std::chrono::high_resolution_clock::now()

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

namespace gpu_bsw_driver{

// for storing the alignment results
struct alignment_results{
  short* g_alAbeg;
  short* g_alBbeg;
  short* g_alAend;
  short* g_alBend;
  short* top_scores;
};

void
kernel_driver_dna(std::vector<std::string> reads, std::vector<std::string> contigs, alignment_results *alignments, short scores[4]);


void
kernel_driver_aa(std::vector<std::string> reads, std::vector<std::string> contigs, alignment_results *alignments, short scoring_matrix[], short openGap, short extendGap);

void
verificationTest(std::string rstFile, short* g_alAbeg, short* g_alBbeg, short* g_alAend,
                 short* g_alBend);
}
#endif
