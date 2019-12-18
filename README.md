# GPU-BSW
License:  

GPU accelerated Smith-Waterman for performing batch alignments (GPU-BSW) Copyright (c) 2019, The
Regents of the University of California, through Lawrence Berkeley National
Laboratory (subject to receipt of any required approvals from the U.S.
Dept. of Energy).  All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative
works, and perform publicly and display publicly, and to permit other to do
so.



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
