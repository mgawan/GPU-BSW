#include "doctest.h"

#include <gpu_bsw/reordering.hpp>

#include <algorithm>
#include <random>
#include <vector>

TEST_CASE("reorder"){
  std::mt19937 gen;
  std::uniform_int_distribution<short> value_dist(0,std::numeric_limits<short>::max());
  std::vector<short> data;
  std::vector<short> ordering;
  std::vector<uint8_t> progress;

  for(int i=0;i<3000;i++){
    const int len = value_dist(gen);
    data.clear();
    ordering.clear();
    for(int i=0;i<len;i++){
      data.push_back(value_dist(gen));
      ordering.push_back(i);
    }

    const auto original = data;

    std::shuffle(ordering.begin(), ordering.end(), gen);

    forward_reorder(data, ordering, progress);
    backward_reorder(data.data(), ordering, progress);

    CHECK(original==data);
  }
}
