#include<utils.hpp>
unsigned getMaxLength (std::vector<std::string> v)
{
  unsigned maxLength = 0;
  for(auto str : v){
    if(maxLength < str.length()){
      maxLength = str.length();
    }
  }
  return maxLength;
}

void initialize_alignments(gpu_bsw_driver::alignment_results *alignments, int max_alignments){
    cudaMallocHost(&(alignments->ref_begin), sizeof(short)*max_alignments);
    cudaMallocHost(&(alignments->ref_end), sizeof(short)*max_alignments);
    cudaMallocHost(&(alignments->query_begin), sizeof(short)*max_alignments);
    cudaMallocHost(&(alignments->query_end), sizeof(short)*max_alignments);
    cudaMallocHost(&(alignments->top_scores), sizeof(short)*max_alignments);
}

void free_alignments(gpu_bsw_driver::alignment_results *alignments){
       cudaErrchk(cudaFreeHost(alignments->ref_begin));
       cudaErrchk(cudaFreeHost(alignments->ref_end));
       cudaErrchk(cudaFreeHost(alignments->query_begin));
       cudaErrchk(cudaFreeHost(alignments->query_end));
       cudaErrchk(cudaFreeHost(alignments->top_scores));

}