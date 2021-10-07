cmake_minimum_required(VERSION 3.5)
project(RM2021_Adv)

find_package( OpenCV REQUIRED )
find_package( Threads)
add_definitions(-w)
set(CMAKE_CXX_STANDARD 11)
include_directories(./misc)
include_directories(./armor) 
include_directories(./dfc) 
add_executable(RM2021_Adv main.cpp )

target_link_libraries( RM2021_Adv ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT} )
