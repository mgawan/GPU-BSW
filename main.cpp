#include "driver.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <string>
#include<bits/stdc++.h>

using namespace std;

void proteinSampleRun(string file){
  vector<string> sequences_a, sequences_b;
  string line_in;

  ifstream file_one(file);
  ifstream file_two(file);

  if(file_one.is_open()){
    while(getline(file_one, line_in)){
      string str_accum;
      if(line_in[0] == '>'){
        continue;
      }else{
        str_accum.append(line_in);
        while(getline(file_one,line_in) && line_in[0] != '>'){
          str_accum.append(line_in);
        }
      }
      if(str_accum.size() <=1024){
      transform(str_accum.begin(), str_accum.end(), str_accum.begin(), ::toupper);
      sequences_a.push_back(str_accum);
      sequences_b.push_back(str_accum);
      }
    }
  }




    short scores_matrix[] = {// 24 x 24 table
   //  A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V   B   Z   X   *
         5, -2, -1, -2, -1, -1, -1,  0, -2, -1, -2, -1, -1, -3, -1,  1,  0, -3, -2,  0, -2, -1, -1, -5,	// A
         -2,  7, -1, -2, -4,  1,  0, -3,  0, -4, -3,  3, -2, -3, -3, -1, -1, -3, -1, -3, -1,  0, -1, -5,	// R
         -1, -1,  7,  2, -2,  0,  0,  0,  1, -3, -4,  0, -2, -4, -2,  1,  0, -4, -2, -3,  5,  0, -1, -5,	// N
         -2, -2,  2,  8, -4,  0,  2, -1, -1, -4, -4, -1, -4, -5, -1,  0, -1, -5, -3, -4,  6,  1, -1, -5,	// D
         -1, -4, -2, -4, 13, -3, -3, -3, -3, -2, -2, -3, -2, -2, -4, -1, -1, -5, -3, -1, -3, -3, -1, -5,	// C
         -1,  1,  0,  0, -3,  7,  2, -2,  1, -3, -2,  2,  0, -4, -1,  0, -1, -1, -1, -3,  0,  4, -1, -5,	// Q
         -1,  0,  0,  2, -3,  2,  6, -3,  0, -4, -3,  1, -2, -3, -1, -1, -1, -3, -2, -3,  1,  5, -1, -5,	// E
         0, -3,  0, -1, -3, -2, -3,  8, -2, -4, -4, -2, -3, -4, -2,  0, -2, -3, -3, -4, -1, -2, -1, -5,	// G
         -2,  0,  1, -1, -3,  1,  0, -2, 10, -4, -3,  0, -1, -1, -2, -1, -2, -3,  2, -4,  0,  0, -1, -5,	// H
         -1, -4, -3, -4, -2, -3, -4, -4, -4,  5,  2, -3,  2,  0, -3, -3, -1, -3, -1,  4, -4, -3, -1, -5,	// I
         -2, -3, -4, -4, -2, -2, -3, -4, -3,  2,  5, -3,  3,  1, -4, -3, -1, -2, -1,  1, -4, -3, -1, -5,	// L
         -1,  3,  0, -1, -3,  2,  1, -2,  0, -3, -3,  6, -2, -4, -1,  0, -1, -3, -2, -3,  0,  1, -1, -5,	// K
         -1, -2, -2, -4, -2,  0, -2, -3, -1,  2,  3, -2,  7,  0, -3, -2, -1, -1,  0,  1, -3, -1, -1, -5,	// M
         -3, -3, -4, -5, -2, -4, -3, -4, -1,  0,  1, -4,  0,  8, -4, -3, -2,  1,  4, -1, -4, -4, -1, -5,	// F
         -1, -3, -2, -1, -4, -1, -1, -2, -2, -3, -4, -1, -3, -4, 10, -1, -1, -4, -3, -3, -2, -1, -1, -5,	// P
         1, -1,  1,  0, -1,  0, -1,  0, -1, -3, -3,  0, -2, -3, -1,  5,  2, -4, -2, -2,  0,  0, -1, -5,	// S
       0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  2,  5, -3, -2,  0,  0, -1, -1, -5, 	// T
         -3, -3, -4, -5, -5, -1, -3, -3, -3, -3, -2, -3, -1,  1, -4, -4, -3, 15,  2, -3, -5, -2, -1, -5, 	// W
         -2, -1, -2, -3, -3, -1, -2, -3,  2, -1, -1, -2,  0,  4, -3, -2, -2,  2,  8, -1, -3, -2, -1, -5, 	// Y
         0, -3, -3, -4, -1, -3, -3, -4, -4,  4,  1, -3,  1, -1, -3, -2,  0, -3, -1,  5, -3, -3, -1, -5, 	// V
         -2, -1,  5,  6, -3,  0,  1, -1,  0, -4, -4,  0, -3, -4, -2,  0,  0, -5, -3, -3,  6,  1, -1, -5, 	// B
         -1,  0,  0,  1, -3,  4,  5, -2,  0, -3, -3,  1, -1, -4, -1,  0, -1, -2, -2, -3,  1,  5, -1, -5, 	// Z
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -5, 	// X
         -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5,  1 	// *
   };

    gpu_bsw_driver::alignment_results results_store;
    gpu_bsw_driver::kernel_driver_aa(sequences_a, sequences_a, &results_store, scores_matrix, -5, -2);
  // cout << "A beg:"<<results_test.g_alAbeg[0]<<"  A end:"<<results_test.g_alAend[0]<<" B beg:"<<results_test.g_alBbeg[0]<<" B end:"<<results_test.g_alBend[0]<<" score:"<<results_test.top_scores[0]<<endl;
  unsigned errors = 0;
    for(unsigned i = 0; i < sequences_a.size(); i++){
      if((results_store.g_alAbeg[i] != results_store.g_alBbeg[i]) || (results_store.g_alAend[i] != results_store.g_alBend[i])){
        errors++;
      }
    }

    cout << "Errors Encountered:"<<errors<<endl;
}


void dnaSampleRun(string refFile, string queFile, string resultFile){
  vector<string> G_sequencesA,
      G_sequencesB;

  string   myInLine;
  ifstream ref_file(refFile);
  ifstream quer_file(queFile);
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


  gpu_bsw_driver::alignment_results results_test;

  short scores[] = {5, -3, -5, -2};

  gpu_bsw_driver::kernel_driver_dna(G_sequencesB, G_sequencesA,&results_test, scores);

  gpu_bsw_driver::verificationTest(resultFile, results_test.g_alAbeg, results_test.g_alBbeg, results_test.g_alAend, results_test.g_alBend);

}

int
main(int argc, char* argv[])
{

 //  proteinSampleRun(argv[1]);
  dnaSampleRun(argv[1], argv[2], argv[3]);


    return 0;
}
