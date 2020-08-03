#ifndef DRIVER_HPP
#define DRIVER_HPP

#include <gpu_bsw/driver.hpp>
#include <gpu_bsw/kernel.hpp>

#include <albp/memory.hpp>
#include <albp/page_locked_fasta.hpp>
#include <albp/page_locked_string.hpp>
#include <albp/read_fasta.hpp>
#include <albp/stream_manager.hpp>
#include <albp/timer.hpp>

#include <iostream>
#include <list>
#include <string>
#include <vector>

// for storing the alignment results
struct AlignmentResults{
  AlignmentResults() = default;
  AlignmentResults(const size_t size);
  albp::cuda_unique_hptr<short> a_begin;
  albp::cuda_unique_hptr<short> b_begin;
  albp::cuda_unique_hptr<short> a_end;
  albp::cuda_unique_hptr<short> b_end;
  albp::cuda_unique_hptr<short> top_scores;
};


struct Scoring {
  short gap_open;
  short gap_extend;
  albp::cuda_unique_dptr<short> d_scoring_matrix;
  albp::cuda_unique_dptr<short> d_encoding_matrix;
};


template<DataType DT>
class StreamConsumer {
 public:
  StreamConsumer(
    const std::shared_ptr<Scoring> scoring,
    AlignmentResults &alignments,
    const albp::PageLockedFastaPair &data,
    const int device_id,
    const size_t max_seq_a_size,
    const size_t max_seq_b_size,
    const size_t max_sequences
  ) : scoring(scoring), alignments(alignments), data(data), device_id(device_id), stream(albp::get_new_stream(device_id)) {
    ALBP_CUDA_ERROR_CHECK(cudaSetDevice(device_id));
    seq_a_gpu       = albp::DeviceMallocUnique<char  >(max_sequences*max_seq_a_size);
    seq_b_gpu       = albp::DeviceMallocUnique<char  >(max_sequences*max_seq_b_size);
    starts_a_gpu    = albp::DeviceMallocUnique<size_t>(max_sequences+1);
    starts_b_gpu    = albp::DeviceMallocUnique<size_t>(max_sequences+1);
    seq_a_start_gpu = albp::DeviceMallocUnique<short >(max_sequences);
    seq_a_end_gpu   = albp::DeviceMallocUnique<short >(max_sequences);
    seq_b_start_gpu = albp::DeviceMallocUnique<short >(max_sequences);
    seq_b_end_gpu   = albp::DeviceMallocUnique<short >(max_sequences);
    scores_gpu      = albp::DeviceMallocUnique<short >(max_sequences);
  }

  void operator()(const albp::RangePair range){
    ALBP_CUDA_ERROR_CHECK(cudaSetDevice(device_id));

    const albp::RangePair range_plus_one(range.begin, range.end+1);

    //Copy from host to device
    albp::copy_to_device_async(starts_a_gpu.get(), data.a.starts.get(), range_plus_one, stream);
    albp::copy_to_device_async(starts_b_gpu.get(), data.b.starts.get(), range_plus_one, stream);

    albp::copy_sequences_to_device_async(seq_a_gpu, data.a, range, stream);
    albp::copy_sequences_to_device_async(seq_b_gpu, data.b, range, stream);

    const auto minSize = std::min(
      albp::get_max_length(data.a, range),
      albp::get_max_length(data.b, range)
    );
    const auto totShmem = 3 * (minSize + 1) * sizeof(short);
    const auto alignmentPad = 4 + (4 - totShmem % 4);
    const auto ShmemBytes = totShmem + alignmentPad;
    if(ShmemBytes > 48000 && DT==DataType::DNA)
      cudaFuncSetAttribute(gpu_bsw::sequence_process_forward_and_reverse<DataType::DNA>, cudaFuncAttributeMaxDynamicSharedMemorySize, ShmemBytes);

    assert(minSize>0);
    assert(ShmemBytes>0);
    gpu_bsw::sequence_process_forward_and_reverse<DT><<<range.size(), minSize, ShmemBytes, stream>>>(
      seq_a_gpu.get(),
      seq_b_gpu.get(),
      starts_a_gpu.get(),
      starts_b_gpu.get(),
      seq_a_start_gpu.get(),
      seq_a_end_gpu.get(),
      seq_b_start_gpu.get(),
      seq_b_end_gpu.get(),
      scores_gpu.get(),
      scoring->gap_open,
      scoring->gap_extend,
      scoring->d_scoring_matrix.get(),
      scoring->d_encoding_matrix.get()
    );
    ALBP_CUDA_ERROR_CHECK(cudaGetLastError());

    copy_to_host_async(alignments.a_begin.get(), seq_a_start_gpu.get(), range, stream);
    copy_to_host_async(alignments.a_end.get(),   seq_a_end_gpu.get(),   range, stream);
    copy_to_host_async(alignments.top_scores.get(), scores_gpu.get(),   range, stream);
  }

 private:
  const int device_id;
  const cudaStream_t stream;
  const std::shared_ptr<Scoring> scoring;
  const albp::PageLockedFastaPair &data;
  albp::cuda_unique_dptr<char>   seq_b_gpu;
  albp::cuda_unique_dptr<char>   seq_a_gpu;
  albp::cuda_unique_dptr<size_t> starts_b_gpu;
  albp::cuda_unique_dptr<size_t> starts_a_gpu;
  albp::cuda_unique_dptr<short>  seq_a_start_gpu;
  albp::cuda_unique_dptr<short>  seq_a_end_gpu;
  albp::cuda_unique_dptr<short>  seq_b_start_gpu;
  albp::cuda_unique_dptr<short>  seq_b_end_gpu;
  albp::cuda_unique_dptr<short>  scores_gpu;
  AlignmentResults &alignments;
};



namespace gpu_bsw_driver{

template<DataType DT>
AlignmentResults kernel_driver(
  const albp::FastaPair &input_data,
  const short scoring_matrix[],
  const short openGap,
  const short extendGap,
  const int streams_per_gpu,
  const size_t chunk_size
){
    albp::Timer timer_total;
    timer_total.start();

    // Assuming that read and contig vectors are same length
    const auto totalAlignments = input_data.sequence_count();

    //This matrix is used only by the RNA kernel
    constexpr short encoding_matrix[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                         0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                         0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                        23,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                         0,0,0,0,0,0,0,0,0,0,20,4,3,6,
                                        13,7,8,9,0,11,10,12,2,0,14,5,
                                         1,15,16,0,19,17,22,18,21};

    int device_count;
    cudaGetDeviceCount(&device_count);

    std::cout<<"Number of available GPUs:"<<device_count<<"\n";

    AlignmentResults alignments(input_data.sequence_count());

    auto pl_fasta = albp::page_lock(input_data);

    //We use a list for streams since a vector would move the StreamConsumers
    //around on allocation, breaking the pointers in stream_function.
    std::list<StreamConsumer<DT>> streams;
    std::vector<albp::StreamFunction> stream_functions;

    for(int device_id=0;device_id<device_count;device_id++){
      ALBP_CUDA_ERROR_CHECK(cudaSetDevice(device_id));

      auto scoring = std::shared_ptr<Scoring>(new Scoring{
        openGap,
        extendGap,
        albp::DeviceMallocUnique<short>(SCORE_MAT_SIZE, scoring_matrix),
        albp::DeviceMallocUnique<short>(ENCOD_MAT_SIZE, encoding_matrix) //Only used by RNA kernel
      });

      for(int s=0;s<streams_per_gpu;s++){
        streams.emplace_back(scoring, alignments, pl_fasta, device_id, input_data.a.maximum_sequence_length, input_data.b.maximum_sequence_length, chunk_size);
        stream_functions.emplace_back(streams.back());
      }
    }

    albp::process_streams(stream_functions, input_data.sequence_count(), chunk_size);

    timer_total.stop();
    std::cout <<"Total Alignments   = "<<totalAlignments<<"\n"
              <<"Max Reference Size = "<<input_data.a.maximum_sequence_length<<"\n"
              <<"Max Query Size     = "<<input_data.b.maximum_sequence_length<<"\n"
              <<"Total Execution Time (seconds) = "<<timer_total.getSeconds() <<std::endl;

  return alignments;
}

}

#endif
