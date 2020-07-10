#include "doctest.h"
#include <gpu_bsw/timer.hpp>

#include <chrono>
#include <thread>

TEST_CASE("Timer Start Stop"){
  Timer timer;
  timer.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  timer.stop();

  CHECK(std::abs(timer.getSeconds()-0.2)<0.01);
}
