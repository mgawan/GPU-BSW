#include <gpu_bsw/alignments.hpp>
#include <gpu_bsw/utils.hpp>

#include <albp/memory.hpp>

gpu_alignments::gpu_alignments(int max_alignments){
    offset_query_gpu = albp::DeviceMalloc<unsigned>(max_alignments);
    offset_ref_gpu   = albp::DeviceMalloc<unsigned>(max_alignments);
    ref_start_gpu    = albp::DeviceMalloc<short>(max_alignments);
    ref_end_gpu      = albp::DeviceMalloc<short>(max_alignments);
    query_end_gpu    = albp::DeviceMalloc<short>(max_alignments);
    query_start_gpu  = albp::DeviceMalloc<short>(max_alignments);
    scores_gpu       = albp::DeviceMalloc<short>(max_alignments);
}

gpu_alignments::~gpu_alignments(){
    ALBP_CUDA_ERROR_CHECK(cudaFree(offset_ref_gpu));
    ALBP_CUDA_ERROR_CHECK(cudaFree(offset_query_gpu));
    ALBP_CUDA_ERROR_CHECK(cudaFree(ref_start_gpu));
    ALBP_CUDA_ERROR_CHECK(cudaFree(ref_end_gpu));
    ALBP_CUDA_ERROR_CHECK(cudaFree(query_start_gpu));
    ALBP_CUDA_ERROR_CHECK(cudaFree(query_end_gpu));
}