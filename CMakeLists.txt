cmake_minimum_required(VERSION 3.20)
project(zfft
  VERSION
    0.1.0
  DESCRIPTION
    "A CUDA C++ library with testsuite and benchmarks"
  LANGUAGES
    CXX CUDA
  )





# cmake-gui
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()


# GoogleTest requires at least C++14
# target_compile_features(mylib PUBLIC cxx_std_17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++17 -Wno-c++17-extensions -Werror")
# TODO: remove --expt-relaxed-constexpr
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Weverything --expt-relaxed-constexpr")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


# pthreads workaround
set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(CUDAToolkit REQUIRED)

include(CheckLanguage)
check_language(CUDA)
cmake_policy(SET CMP0135 NEW)

enable_testing()
add_subdirectory(src)
add_subdirectory(test)
#add_subdirectory(bench)