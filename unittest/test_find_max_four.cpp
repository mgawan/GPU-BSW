#include "doctest.h"

#include <gpu_bsw/kernel.hpp>

#include <algorithm>
#include <limits>
#include <random>
#include <vector>

TEST_MAIN("findMaxFour"){
  std::mt19937 gen;
  std::uniform_int_distribution<short> value_dist(std::numeric_limits<short>::min(),std::numeric_limits<short>::max());
  std::array<short,4> dat;

  for(int i=0;i<100;i++){
    for(int i=0;i<4;i++){
      dat.at(i) = value_dist(gen);
    }

    const auto fmf_max = findMaxFour(dat.at(0), dat.at(1), dat.at(2), dat.at(3));
    const auto correct_max = *std::max_element(dat.begin(),dat.end());
    CHECK(fmf_max==correct_max);
  }
}
