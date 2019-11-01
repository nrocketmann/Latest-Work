cmake_minimum_required(VERSION 3.13)
project(FirstLibrary)

set(CMAKE_CXX_STANDARD 14)

add_executable(FirstLibrary main.cpp Model.cpp Model.h Operation.cpp Operation.h Loss.cpp Loss.h)