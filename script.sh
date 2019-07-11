#!/bin/bash


module load cuda
export OMP_NUM_THREADS=1
srun nvprof --kernels "align_sequences_gpu" --metrics inst_integer,flop_count_sp,inst_fp_64,inst_fp_32,flop_count_dp,gld_transactions,gst_transactions,dram_read_transactions,dram_write_transactions,l2_read_transactions,l2_write_transactions,shared_load_transactions,shared_store_transactions ./program_gpu ./test_data/ref_file_30000.txt ./test_data/que_file_30000.txt ./test_data/results_30000 >> prfile_results.txt
