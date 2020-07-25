#include <gpu_bsw/read_fasta.hpp>

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

/*
  Reads FASTA files and fills the corresponding buffers.
  FASTA files contain sequences that are usually on separate lines.
  The file reader detects a '>' then concatenates all the following lines into one sequence, until the next '>' or EOF.
  See more about FASTA format : https://en.wikipedia.org/wiki/FASTA_format
*/
FastaInput ReadFasta(const std::string &filename){
  std::ifstream fin(filename);
  if(!fin.good()){
    throw std::runtime_error("Failed to open file '"+filename+"'!");
  }

  FastaInput fasta;

  bool reading_sequence = false; //Whether we are presently reading a sequence

  const std::string line_starts = "></+";
  /* The information of reverse-complementing is simulated by changing the first character of the sequence.
   * This is not explicitly FASTA-compliant, although regular FASTA files will simply be interpreted as Forward-Natural direction.
   * From the header of every sequence:
   * - '>' translates to 0b00 (0) = Forward, natural
   * - '<' translates to 0b01 (1) = Reverse, natural
   * - '/' translates to 0b10 (2) = Forward, complemented
   * - '+' translates to 0b11 (3) = Reverse, complemented
   * No protection is done, so any other number will only have its two first bytes counted as above.
   */

  //Load sequences from the files
  std::string input_line;
  while (std::getline(fin, input_line)){
    //Skip empty lines
    if(input_line.empty())
      continue;
    //Skip lines entirely made of whitespace
    if(std::all_of(input_line.begin(), input_line.end(), [](const auto &x) {return std::isspace(x);}))
      continue;

    //Determine if the line starts with a special character
    const auto q = std::find(line_starts.begin(), line_starts.end(), input_line.front());

    if(q!=line_starts.end()){                        //Line begins with a special character. It's the start of a new sequence!
      fasta.modifiers.push_back(*q);                 //Make a note of which special modifying character was used
      fasta.headers.push_back(input_line.substr(1)); //Copy the header text, dropping the start/mod character

      if (reading_sequence) {
        // a sequence was already being read. Now it's done, so we should find its length.
        fasta.total_sequence_bytes += fasta.sequences.back().length();
        fasta.maximum_sequence_length = std::max(fasta.sequences.back().length(), fasta.maximum_sequence_length);
      }
      reading_sequence = false;

    } else if (!reading_sequence) {
      //If we're here then the line didn't begin with a special character and
      //we're not currently appending to a sequence, so we start a new sequence.
      fasta.sequences.push_back(input_line);
      reading_sequence = true;
    } else if (reading_sequence) {
      fasta.sequences.back() += input_line;
    }
  }

  //We've reached the end of the file, so we need one last update of the sequence lengths
  fasta.total_sequence_bytes += fasta.sequences.back().length();
  fasta.maximum_sequence_length = std::max(fasta.sequences.back().length(), fasta.maximum_sequence_length);

  return fasta;
}



FastaPair ReadFastaQueryTargetPair(const std::string &query, const std::string &target){
  FastaPair ret;
  ret.a = ReadFasta(query);
  ret.b = ReadFasta(target);

  if(ret.a.sequences.size()!=ret.b.sequences.size())
    throw std::runtime_error("Query and Target files were not the same length!");

  return ret;
}


size_t FastaInput::sequence_count() const {
  return sequences.size();
}

uint64_t FastaPair::total_cells_1_to_1() const {
  uint64_t count = 0;
  for(size_t i=0;i<a.sequence_count();i++){
    count += a.sequences.at(i).size()*b.sequences.at(i).size();
  }
  return count;
}

size_t FastaPair::sequence_count() const {
  assert(a.sequence_count()==b.sequence_count());
  return a.sequence_count();
}



SortUnsortFastaPair::SortUnsortFastaPair(FastaPair &fp){
  //Load the ordering
  ordering.reserve(fp.sequence_count());
  for(size_t i=0;i<fp.sequence_count();i++)
    ordering.push_back(i);

  std::sort(ordering.begin(), ordering.end(), [&](const size_t ai, const size_t bi){
    return  std::min(fp.a.sequences.at(ai).size(), fp.b.sequences.at(ai).size())
            <std::min(fp.a.sequences.at(bi).size(), fp.b.sequences.at(bi).size());
  });

  forward_reorder(fp.a.sequences, ordering, visited);
  // forward_reorder(fp.a.headers,   ordering, visited); //TODO: We don't use these just now
  // forward_reorder(fp.a.modifiers, ordering, visited);

  forward_reorder(fp.b.sequences, ordering, visited);
  // forward_reorder(fp.b.headers,   ordering, visited);
  // forward_reorder(fp.b.modifiers, ordering, visited);
}


void SortUnsortFastaPair::unsort(FastaPair &fp) const {
    backward_reorder(fp.a.sequences.data(), ordering, visited);
    // backward_reorder(fp.a.headers.data(),   ordering, visited); //TODO: We don't use these just now
    // backward_reorder(fp.a.modifiers.data(), ordering, visited);

    backward_reorder(fp.b.sequences.data(), ordering, visited);
    // backward_reorder(fp.b.headers.data(),   ordering, visited);
    // backward_reorder(fp.b.modifiers.data(), ordering, visited);
  }