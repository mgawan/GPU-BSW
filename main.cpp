#include "driver.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <string>


using namespace std;
using auto_timer_t = tim::auto_timer;

int
main(int argc, char* argv[])
{
  tim::timemory_init(argc, argv);
//  omp_set_num_threads(1);
//  cudaSetDevice(0);
    vector<string> G_sequencesA, G_sequencesB;

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
    cout << "total alignments:" << G_sequencesB.size() << endl;

    using auto_timer_list_type = typename auto_timer_t::component_type::list_type;
    auto _orig_init = auto_timer_list_type::get_initializer();
    auto_timer_list_type::get_initializer() = [=](auto_timer_list_type& al)
    {
        using namespace tim::component;
        _orig_init(al);
        tim::settings::instruction_roofline() = true;
        al.init<gpu_roofline_sp_flops>();
    };

    auto_timer_t main(argv[0]);

    callAlignKernel(G_sequencesB, G_sequencesA, largestB, largestA, G_sequencesA.size(),
                    &g_alAbeg, &g_alBbeg, &g_alAend, &g_alBend, argv[3]);

  main.stop();
    verificationTest(argv[3], g_alAbeg, g_alBbeg, g_alAend, g_alBend);

    tim::timemory_finalize();
    
    return 0;
}
