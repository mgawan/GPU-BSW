#ifndef DRIVER_HPP
#define DRIVER_HPP

#include "kernel.hpp"
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

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
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if(abort)
            exit(code);
    }
}

void
callAlignKernel(std::vector<std::string> reads, std::vector<std::string> contigs,
                unsigned maxReadSize, unsigned maxContigSize, unsigned totalAlignments,
                short** gg_alAbeg, short** gg_alBbeg, short** gg_alAend,
                short** gg_alBend, char* rstFile);

void
verificationTest(char* rstFile, short* g_alAbeg, short* g_alBbeg, short* g_alAend,
                 short* g_alBend);

#endif
