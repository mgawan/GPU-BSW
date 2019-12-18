# gpu-sw-timemory

To Build:\s
mkdir build \s
cd build \s
cmake CMAKE_BUILD_TYPE=Release .. \s
make \s

\s
To Execute: \s
export OMP_NUM_THREADS=<number of GPUs available>
./program_gpu ../test-data/ref_file_30000.txt ../test-data/que_file_30000.txt ../test-data/results_30000

Contact: mgawan@lbl.gov
