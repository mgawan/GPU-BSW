#include "driver.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <string>

using namespace std;

int
main(int argc, char* argv[])
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cudaDeviceProp prop[deviceCount];
    for(int i = 0; i < deviceCount; i++)
        cudaGetDeviceProperties(&prop[i], 0);

    for(int i = 0; i < deviceCount; i++)
    {
        cout << "total Global Memory available on Device: " << i
             << " is:" << prop[i].totalGlobalMem << endl;
    }

    vector<string> G_sequencesA,
        G_sequencesB;

    string   myInLine;
    ifstream ref_file(argv[1]);
    ifstream quer_file(argv[2]);
    unsigned largestA = 0, largestB = 0;

    int totSizeA = 0, totSizeB = 0;
    if(ref_file.is_open())
    {
        while(getline(ref_file, myInLine))
        {
            string seq = myInLine.substr(myInLine.find(":") + 1, myInLine.size() - 1);
            G_sequencesA.push_back(seq);
            totSizeA += seq.size();
            if(seq.size() > largestA)
            {
                largestA = seq.size();
            }
        }
    }

    if(quer_file.is_open())
    {
        while(getline(quer_file, myInLine))
        {
            string seq = myInLine.substr(myInLine.find(":") + 1, myInLine.size() - 1);
            G_sequencesB.push_back(seq);
            totSizeB += seq.size();
            if(seq.size() > largestB)
            {
                largestB = seq.size();
            }
        }
    }

    short* g_alAbeg;
    short* g_alBbeg;
    short* g_alAend;
    short* g_alBend;

    callAlignKernel(G_sequencesB, G_sequencesA, largestB, largestA, G_sequencesA.size(),
                    &g_alAbeg, &g_alBbeg, &g_alAend, &g_alBend, argv[3]);

    // cout <<"start ref:"<<g_alAbeg[0]<<" end ref:"<<g_alAend[0]<<endl;
    // cout <<"start que:"<<g_alBbeg[0]<<" end que:"<<g_alBend[0]<<endl;
    // cout <<"start ref:"<<g_alAbeg[1]<<" end ref:"<<g_alAend[1]<<endl;
    // cout <<"start que:"<<g_alBbeg[1]<<" end que:"<<g_alBend[1]<<endl;
    verificationTest(argv[3], g_alAbeg, g_alBbeg, g_alAend, g_alBend);

    return 0;
}
