#include <gpu_bsw/driver.hpp>

#include <albp/ranges.hpp>

#include <algorithm>

AlignmentResults::AlignmentResults(const size_t size){
  a_begin     = albp::PageLockedMallocUnique<short>(size);
  a_end       = albp::PageLockedMallocUnique<short>(size);
  b_begin     = albp::PageLockedMallocUnique<short>(size);
  b_end       = albp::PageLockedMallocUnique<short>(size);
  top_scores  = albp::PageLockedMallocUnique<short>(size);
}



int get_new_min_length(const short *const alAend, const short *const alBend, const albp::RangePair &rp){
  short maxA = 0;
  short maxB = 0;
  for(auto i = rp.begin; i < rp.end; i++){
    maxA = std::max(maxA, alAend[i]);
    maxB = std::max(maxB, alBend[i]);
  }
  return std::min(maxA, maxB);
}