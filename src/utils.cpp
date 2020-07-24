#include <algorithm>
#include <gpu_bsw/utils.hpp>

size_t getMaxLength(const std::vector<std::string> &v){
  const auto maxi = std::max_element(v.begin(), v.end(),
    [](const auto &a, const auto &b) { return a.size()<b.size(); }
  );
  return maxi->size();
}



int get_new_min_length(const short *const alAend, const short *const alBend, const int blocksLaunched){
  short maxA = 0;
  short maxB = 0;
  for(int i = 0; i < blocksLaunched; i++){
    maxA = std::max(maxA, alAend[i]);
    maxB = std::max(maxB, alBend[i]);
  }
  return std::min(maxA, maxB);
}