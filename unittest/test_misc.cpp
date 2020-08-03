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