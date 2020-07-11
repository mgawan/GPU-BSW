#include "doctest.h"
#include <gpu_bsw/page_locked_string.hpp>

#include <stdexcept>

TEST_CASE("PageLockedString"){
  PageLockedString pls(20);
  CHECK(pls.capacity()==20);
  CHECK(pls.empty());
  CHECK(!pls.full());
  CHECK(pls.size()==0);
  CHECK(pls.size_left()==20);
  CHECK(pls.str()=="");

  pls += "testing";
  CHECK(pls.capacity()==20);
  CHECK(!pls.empty());
  CHECK(!pls.full());
  CHECK(pls.size()==7);
  CHECK(pls.size_left()==13);
  CHECK(pls.str()=="testing");

  pls += "andagain";
  CHECK(pls.capacity()==20);
  CHECK(!pls.empty());
  CHECK(!pls.full());
  CHECK(pls.size()==15);
  CHECK(pls.size_left()==5);
  CHECK(pls.str()=="testingandagain");

  CHECK_THROWS_AS(pls += "toolong", std::runtime_error);
  CHECK(pls.capacity()==20);
  CHECK(!pls.empty());
  CHECK(!pls.full());
  CHECK(pls.size()==15);
  CHECK(pls.size_left()==5);
  CHECK(pls.str()=="testingandagain");

  pls += "done.";
  CHECK(pls.capacity()==20);
  CHECK(!pls.empty());
  CHECK(pls.full());
  CHECK(pls.size()==20);
  CHECK(pls.size_left()==0);
  CHECK(pls.str()=="testingandagaindone.");

  pls.clear();
  CHECK(pls.capacity()==20);
  CHECK(pls.empty());
  CHECK(!pls.full());
  CHECK(pls.size()==0);
  CHECK(pls.size_left()==20);
  CHECK(pls.str()=="");
}