#include "doctest.h"
#include <gpu_bsw/read_fasta.hpp>
#include <gpu_bsw/timer.hpp>
#include <gpu_bsw/utils.hpp>

#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>

template<class T>
std::ostream& operator<<(std::ostream &out, std::vector<T> &vec){
  for(const auto &x: vec)
    out<<x<<" ";
  return out;
}

TEST_CASE("No fasta file"){
  CHECK_THROWS_AS(ReadFasta("not-a-file"), std::runtime_error);
}

//TODO: Test for modifiers reading correctly
TEST_CASE("Read Fasta"){
  FastaInput fasta;
  CHECK_NOTHROW(fasta=ReadFasta("../test-data/dna-reference.fasta"));

  CHECK(fasta.sequences.size()==30'000);
  CHECK(fasta.modifiers.size()==30'000);
  CHECK(fasta.headers.size()==30'000);
  CHECK(fasta.maximum_sequence_length==579);
  CHECK(getMaxLength(fasta.sequences)==579);
  CHECK(fasta.sequence_count()==30'000);
  CHECK(fasta.sequences.at(5)=="CGCACAAATCAGAAGCTCCGGGTGGCAAACACAGCTAAATAGTTGTAATTATGGAATATAGAAAAATGTTCGATTGTCGTTATGAGGATTATGAGCGCCTCAAAGCCCCCCCACCGCAAAAAGGCCCTGTGTTCGCCCCTCTCCACCCATCCATCGCATGGCCCAACGAAGCGGATATCGCTCCGGAATCCTCCTACGAAAAACTTCTGTAAAAAGAACAAAACCGGAAATCCACTTGGGAACGCGAAACCCCAGCTTCGCATATTGACCCAGAAGATCAACAGTAGAATTTGTGGCAACGGAACAACGTCCCGGAACTTCTCCTGAACCAAAACAACTTCACTGTTCGATTCCCCGCACCATTACATGATGCAGCGTTCCCGGTGTGTCAAGTCTCGCTCCTCGTGGCATATGGCTCTCTTGTCTTTTGCTTTTCAAAAGCTGCCTGCACAAATCGTTTATTCCTCACTGCAAAATACAATTTTCTACGCTATTGCACTGCGTCCCCTCAGGCTCACTCTCAGGCTCAATAATGACAGAAAATTCAGCGGTAAATGGATGGAATCATACGTATGTGAA");
}

TEST_CASE("Read Pair"){
  const auto input_data = ReadFastaQueryTargetPair("../test-data/dna-reference.fasta", "../test-data/dna-query.fasta");
  CHECK(input_data.sequence_count()==30'000);
}

TEST_CASE("Reorder Fasta"){
  std::mt19937 gen;
  std::uniform_int_distribution<short> value_dist(0,std::numeric_limits<short>::max());

  auto input_data = ReadFastaQueryTargetPair("../test-data/dna-reference.fasta", "../test-data/dna-query.fasta");

  const auto original = input_data;

  Timer timer_sort;
  timer_sort.start();
  SortUnsortFastaPair sufp(input_data);
  timer_sort.stop();
  std::cerr<<"t Sort time = "<<timer_sort.getSeconds()<<" s"<<std::endl;

  Timer timer_unsort;
  timer_unsort.start();
  sufp.unsort(input_data);
  timer_unsort.stop();
  std::cerr<<"t Unsort time = "<<timer_unsort.getSeconds()<<" s"<<std::endl;

  CHECK(original.a.headers==input_data.a.headers);
  CHECK(original.a.sequences==input_data.a.sequences);
  CHECK(original.a.modifiers==input_data.a.modifiers);

  CHECK(original.b.headers==input_data.b.headers);
  CHECK(original.b.sequences==input_data.b.sequences);
  CHECK(original.b.modifiers==input_data.b.modifiers);
}

TEST_CASE("Reorder Fasta Single Unsort"){
  std::mt19937 gen;
  std::uniform_int_distribution<short> value_dist(0,std::numeric_limits<short>::max());

  auto input_data = ReadFastaQueryTargetPair("../test-data/dna-reference.fasta", "../test-data/dna-query.fasta");

  const auto original = input_data;

  Timer timer_sort;
  timer_sort.start();
  SortUnsortFastaPair sufp(input_data);
  timer_sort.stop();
  std::cerr<<"t Sort time = "<<timer_sort.getSeconds()<<" s"<<std::endl;

  Timer timer_unsort;
  timer_unsort.start();
  sufp.unsort(input_data.a.modifiers.data());
  timer_unsort.stop();
  std::cerr<<"t Unsort time = "<<timer_unsort.getSeconds()<<" s"<<std::endl;

  CHECK(original.a.modifiers==input_data.a.modifiers);
}