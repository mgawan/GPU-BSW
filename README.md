# GPU-BSW
**License:**  
        
**GPU accelerated Smith-Waterman for performing batch alignments (GPU-BSW) Copyright (c) 2019, The
Regents of the University of California, through Lawrence Berkeley National
Laboratory (subject to receipt of any required approvals from the U.S.
Dept. of Energy).  All rights reserved.**

**If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.**

**NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative
works, and perform publicly and display publicly, and to permit other to do
so.**
       


To Build:

    mkdir build 
    cd build 
    cmake CMAKE_BUILD_TYPE=Release .. 
    make 

To Execute DNA test run:

    ./program_gpu dna ../test-data/dna-reference.fasta ../test-data/dna-query.fasta output

To Execute Protein test run:

    ./program_gpu aa ../test-data/protein-reference.fasta ../test-data/protein-query.fasta output

Contact: mgawan@lbl.gov

If you use GPU-BSW in your project, please cite this repo as:  

*Muaaz G. Awan, GPU accelerated Smith-Waterman for performing batch alignments (GPU-BSW), 2019, Github Repository: https://github.com/m-gul/GPU-BSW/*
