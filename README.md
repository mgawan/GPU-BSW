# gpu-sw-timemory

To Build:<br />
mkdir build <br />
cd build <br />
cmake CMAKE_BUILD_TYPE=Release .. <br />
make <br />

<br />
To Execute: <br />
export OMP_NUM_THREADS=number of GPUs available <br />
./program_gpu ../test-data/ref_file_30000.txt ../test-data/que_file_30000.txt ../test-data/results_30000 <br />

<br />
Contact: mgawan@lbl.gov
