#include <gpu_bsw/alignments.hpp>
#include <gpu_bsw/utils.hpp>

gpu_alignments::gpu_alignments(int max_alignments){
    offset_query_gpu = DeviceMalloc<unsigned>(max_alignments);
    offset_ref_gpu   = DeviceMalloc<unsigned>(max_alignments);
    ref_start_gpu    = DeviceMalloc<short>(max_alignments);
    ref_end_gpu      = DeviceMalloc<short>(max_alignments);
    query_end_gpu    = DeviceMalloc<short>(max_alignments);
    query_start_gpu  = DeviceMalloc<short>(max_alignments);
    scores_gpu       = DeviceMalloc<short>(max_alignments);
}

gpu_alignments::~gpu_alignments(){
    cudaErrchk(cudaFree(offset_ref_gpu));
    cudaErrchk(cudaFree(offset_query_gpu));
    cudaErrchk(cudaFree(ref_start_gpu));
    cudaErrchk(cudaFree(ref_end_gpu));
    cudaErrchk(cudaFree(query_start_gpu));
    cudaErrchk(cudaFree(query_end_gpu));
}