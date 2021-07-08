#include "driver.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <string>

constexpr int MAX_REF_LEN    =      1200;
constexpr int MAX_QUERY_LEN  =       300;
constexpr int GPU_ID         =         0;

constexpr unsigned int DATA_SIZE = std::numeric_limits<unsigned int>::max();;

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

  //   vector<string> G_sequencesA,
  //       G_sequencesB;

  //   string refFile = argv[1];
  // string queFile = argv[2];
  // string out_file = argv[3];
  // string res_file = argv[4];

  //   string   myInLine;
  //   ifstream ref_file(refFile);
  //   ifstream quer_file(queFile);
  //   unsigned largestA = 0, largestB = 0;

  //   int totSizeA = 0, totSizeB = 0;
  //   if(ref_file.is_open())
  //   {
  //       while(getline(ref_file, myInLine))
  //       {
  //           string seq = myInLine.substr(myInLine.find(":") + 1, myInLine.size() - 1);
  //           G_sequencesA.push_back(seq);
  //           totSizeA += seq.size();
  //           if(seq.size() > largestA)
  //           {
  //               largestA = seq.size();
  //           }
  //       }
  //   }

  //   if(quer_file.is_open())
  //   {
  //       while(getline(quer_file, myInLine))
  //       {
  //           string seq = myInLine.substr(myInLine.find(":") + 1, myInLine.size() - 1);
  //           G_sequencesB.push_back(seq);
  //           totSizeB += seq.size();
  //           if(seq.size() > largestB)
  //           {
  //               largestB = seq.size();
  //           }
  //       }
  //   }

string refFile = argv[1];
  string queFile = argv[2];
  string out_file = argv[3];
  string res_file = argv[4];

  vector<string> ref_sequences, que_sequences;
  string   lineR, lineQ;

  ifstream ref_file(refFile);
  ifstream quer_file(queFile);

  unsigned largestA = 0, largestB = 0;

  int totSizeA = 0, totSizeB = 0;

  // extract reference sequences
  if(ref_file.is_open() && quer_file.is_open())
  {
    while(getline(ref_file, lineR))
    {
      getline(quer_file, lineQ);
      if(lineR[0] == '>')
      {
        if (lineR[0] == '>')
          continue;
        else
        {
          std::cout << "FATAL: Mismatch in lines" << std::endl;
          exit(-2);
        }
      }
      else
      {
        if (lineR.length() <= MAX_REF_LEN && lineQ.length() <= MAX_QUERY_LEN)
        {
          ref_sequences.push_back(lineR);
          que_sequences.push_back(lineQ);

          totSizeA += lineR.length();
          totSizeB += lineQ.length();

          if(lineR.length() > largestA)
            largestA = lineR.length();

          if(lineQ.length() > largestA)
            largestB = lineQ.length();
        }
      }
      if (ref_sequences.size() == DATA_SIZE)
          break;
    }

    ref_file.close();
    quer_file.close();
  }



    short* g_alAbeg;
    short* g_alBbeg;
    short* g_alAend;
    short* g_alBend;
    short* g_scores;

    callAlignKernel(que_sequences, ref_sequences, largestB, largestA, ref_sequences.size(),
                    &g_alAbeg, &g_alBbeg, &g_alAend, &g_alBend, &g_scores, argv[4]);

    cudaDeviceSynchronize();
    ofstream results_file(out_file);

    std::cout << std::endl << "STATUS: Writing results..." << std::endl;

    // write the results header
    results_file << "alignment_scores\t"     << "reference_begin_location\t" << "reference_end_location\t" 
                << "query_begin_location\t" << "query_end_location"         << std::endl;
    for(int i = 0; i < ref_sequences.size(); i++){
        results_file <<g_scores[i]<<"\t"<<g_alAbeg[i]-1<<"\t"<<g_alAend[i]-1<<"\t"<<g_alBbeg[i]-1<<"\t"<<g_alBend[i]-1<<endl;
    }
    //verificationTest(argv[4], g_alAbeg, g_alBbeg, g_alAend, g_alBend);

    return 0;
}
