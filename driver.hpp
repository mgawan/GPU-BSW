#ifndef DRIVER_HPP
#define DRIVER_HPP

#include "kernel.hpp"
#include <chrono>
#include <cmath>
#include <string>
#include <vector>
#include <omp.h>
#include "alignments.hpp"

#define NSTREAMS 2

#define NOW std::chrono::high_resolution_clock::now()



namespace gpu_bsw_driver{

// for storing the alignment results
struct alignment_results{
  short* ref_begin = nullptr;
  short* query_begin = nullptr;
  short* ref_end = nullptr;
  short* query_end = nullptr;
  short* top_scores = nullptr;
};



void kernel_driver_dna(
  const std::vector<std::string> &reads,
  const std::vector<std::string> &contigs,
  gpu_bsw_driver::alignment_results *const alignments,
  const short scores[4]
);



void kernel_driver_aa(
  const std::vector<std::string> &reads,
  const std::vector<std::string> &contigs,
  gpu_bsw_driver::alignment_results *const alignments,
  const short scoring_matrix[],
  const short openGap,
  const short extendGap
);

void
verificationTest(std::string rstFile, short* g_alAbeg, short* g_alBbeg, short* g_alAend,
                 short* g_alBend);
}
#endif
