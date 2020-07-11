#pragma once

#include <stdexcept>

#include "utils.hpp"

class PageLockedString {
  public:
    PageLockedString(size_t capacity) : _str(PageLockedMalloc<char>(capacity)), _capacity(capacity) {}

    ~PageLockedString(){
      if(_str)
        cudaErrchk(cudaFreeHost(_str));
    }

    PageLockedString& operator+=(const std::string &o){
      if(_size+o.size()>_capacity)
        throw std::runtime_error("Appending to the PageLockedString would go above its capacity!");
      memcpy(&_str[_size], o.c_str(), o.size());
      _size += o.size();
      return *this;
    }

    char* data()      const { return _str;  }
    size_t size()     const { return _size; }
    size_t size_left()const { return _capacity-_size; }
    bool empty()      const { return _size==0; }
    bool full()       const { return _size==_capacity; }
    std::string str() const { return std::string(_str, _str+_size); }
    size_t capacity() const { return _capacity; }
    void clear() { _size=0; }

  private:
    char *const _str = nullptr;
    const size_t _capacity = 0;
    size_t _size = 0;
};