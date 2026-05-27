cmake_minimum_required(VERSION 3.18)
project(toploc_search)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(faiss REQUIRED)

pybind11_add_module(toploc_search toploc_search.cpp)

target_link_libraries(toploc_search PRIVATE faiss)