#ifndef ALIGNMENTS_HPP
#define ALIGNMENTS_HPP

class gpu_alignments{
  public:
    short*    ref_start_gpu    = nullptr;
    short*    ref_end_gpu      = nullptr;
    short*    query_start_gpu  = nullptr;
    short*    query_end_gpu    = nullptr;
    short*    scores_gpu       = nullptr;
    unsigned* offset_ref_gpu   = nullptr;
    unsigned* offset_query_gpu = nullptr;

    gpu_alignments(int max_alignments);
    ~gpu_alignments();
};


#endif