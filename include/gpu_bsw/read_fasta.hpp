#pragma once

#include <gpu_bsw/reordering.hpp>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>


struct FastaInput {
  std::vector<std::string> sequences;
  std::vector<std::string> headers;
  std::vector<uint8_t>     modifiers;
  size_t maximum_sequence_length = 0;
  size_t total_sequence_bytes    = 0;
  size_t sequence_count() const;
};

struct FastaPair {
  FastaInput a;
  FastaInput b;
  uint64_t total_cells_1_to_1() const;
  size_t sequence_count() const;
};

FastaInput ReadFasta(const std::string &filename);

FastaPair ReadFastaQueryTargetPair(const std::string &query, const std::string &target);



class SortUnsortFastaPair {
 public:
  SortUnsortFastaPair(FastaPair &fp);

  template<class T>
  void unsort(T *data) const {
    backward_reorder(data, ordering, visited);
  }

  void unsort(FastaPair &fp) const;

 private:
  std::vector<int32_t> ordering;
  mutable std::vector<uint8_t> visited;
};