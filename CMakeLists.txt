cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)
project(MyCode)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(MyCode MyCode.cxx )
target_link_libraries(MyCode ${ITK_LIBRARIES})