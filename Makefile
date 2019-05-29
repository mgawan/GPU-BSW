program_debug: main.cu
	nvcc -std=c++11 -g -G -arch=compute_70 -code=sm_70 main.cu -o program_debug

program_gpu: main.cu
	nvcc -std=c++11 -O3 -arch=compute_70 -code=sm_70 main.cu -o program_gpu
