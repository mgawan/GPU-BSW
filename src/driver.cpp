#include <gpu_bsw/driver.hpp>

AlignmentResults::AlignmentResults(const size_t size){
  a_begin     = albp::PageLockedMallocUnique<short>(size);
  a_end       = albp::PageLockedMallocUnique<short>(size);
  b_begin     = albp::PageLockedMallocUnique<short>(size);
  b_end       = albp::PageLockedMallocUnique<short>(size);
  top_scores  = albp::PageLockedMallocUnique<short>(size);
}
