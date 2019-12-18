# gpu-sw-timemory

To Build:
mkdir build
cd build
cmake CMAKE_BUILD_TYPE=Release ..
make

To Execute:
export OMP_NUM_THREADS=<number of GPUs available>
./program_gpu ../test-data/ref_file_30000.txt ../test-data/que_file_30000.txt ../test-data/results_30000

Contact: mgawan@lbl.gov
