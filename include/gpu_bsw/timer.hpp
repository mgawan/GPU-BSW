#pragma once

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>

class Timer {
 public:
  Timer() = default;
  Timer(const std::string &name) : name(name) {}

  void clear() {elapsed_time = std::chrono::microseconds(0);}
  void start() {
    if(started)
      throw std::runtime_error("Timer was not stopped!");

    start_time = std::chrono::high_resolution_clock::now();
    started = true;
  }
  void restart() { clear(); start(); }

  void stop() {
    if(!started)
      throw std::runtime_error("Timer was not started!");

    const auto this_duration = std::chrono::high_resolution_clock::now() - start_time;
    elapsed_time += std::chrono::duration_cast<std::chrono::microseconds>(this_duration);
    started = false;
  }

  void print() const {
    std::cout << name << " = " << elapsed_time.count() << " msec"   << std::endl;
  }

  double getTime() const { return elapsed_time.count();}

  double getSeconds() const { return elapsed_time.count()/(1e6); }

 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
  const std::string name;
  std::chrono::microseconds elapsed_time = std::chrono::microseconds(0);
  bool started = false;
};
