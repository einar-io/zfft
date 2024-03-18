find_package(GTest REQUIRED)

add_executable(${PROJECT_NAME}test)

set_target_properties(${PROJECT_NAME}test
  PROPERTIES OUTPUT_NAME
    run_${PROJECT_NAME}test
)
set_target_properties(${PROJECT_NAME}test PROPERTIES CUDA_ARCHITECTURES 50)

target_sources(${PROJECT_NAME}test
  PRIVATE
    tests.cu
)

target_compile_features(${PROJECT_NAME}test PRIVATE cxx_std_20)



target_link_libraries(${PROJECT_NAME}test zfft)

target_link_libraries(${PROJECT_NAME}test GTest::gtest_main)

target_link_libraries(${PROJECT_NAME}test ${CONAN_LIBS})

include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME}test)
