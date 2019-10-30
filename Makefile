objects = main.o kernel.o driver.o
ARCH = compute_70
ifeq ($(DEBUG),TRUE)
	NVCCFLAGS = -g -G -Xcompiler -fopenmp
else
	NVCCFLAGS = -O3 -Xcompiler -fopenmp
endif

program_gpu: $(objects)
	nvcc -std=c++11 $(NVCCFLAGS) -arch=$(ARCH) $(objects) -o program_gpu
main.o: main.cpp driver.hpp
	nvcc -std=c++11 -x cu $(NVCCFLAGS) -arch=$(ARCH) -I. -c main.cpp -o $@
driver.o: driver.cpp driver.hpp kernel.hpp
	nvcc -std=c++11 -x cu $(NVCCFLAGS) -arch=$(ARCH) -I. -c driver.cpp -o $@
kernel.o: kernel.cpp kernel.hpp
	nvcc -std=c++11 -x cu $(NVCCFLAGS) -arch=$(ARCH) -I. -c kernel.cpp -o $@
clean:
	rm *.o program_gpu
	echo "all object and executables deleted"
