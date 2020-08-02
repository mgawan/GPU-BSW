#include "doctest.h"

#include <gpu_bsw/driver.hpp>
#include <gpu_bsw/kernel.hpp>

#include <albp/ranges.hpp>

#include <vector>

TEST_CASE("swap"){
  int a = 3;
  int b = 5;
  gpu_bsw::swap(a,b);
  CHECK(a==5);
  CHECK(b==3);
}

TEST_CASE("get_new_min_length"){
  const std::vector<short> a = {{-29319,   5258,  -8504, -29051,  21807,  10263,   9175,  -5258, 30639, -29876}};
  const std::vector<short> b = {{29449, -13061,  -8505,  -8933,  -8052,  19742, -20964, -15165, 29682, -16370}};

  CHECK(get_new_min_length(a.data(), b.data(), albp::RangePair(0,a.size()))==29682);
}