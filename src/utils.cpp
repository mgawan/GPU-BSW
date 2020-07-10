#include <algorithm>
#include <gpu_bsw/utils.hpp>

size_t getMaxLength(const std::vector<std::string> &v){
  const auto maxi = std::max_element(v.begin(), v.end(),
    [](const auto &a, const auto &b) { return a.size()<b.size(); }
  );
  return maxi->size();
}



int get_new_min_length(short* alAend, short* alBend, int blocksLaunched){
        int newMin = 1000;
        int maxA = 0;
        int maxB = 0;
        for(int i = 0; i < blocksLaunched; i++){
          if(alBend[i] > maxB ){
              maxB = alBend[i];
          }
          if(alAend[i] > maxA){
            maxA = alAend[i];
          }
        }
        newMin = (maxB > maxA)? maxA : maxB;
        return newMin;
}