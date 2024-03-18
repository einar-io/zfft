add_library( ${PROJECT_NAME} SHARED )

target_sources(${PROJECT_NAME}
  PRIVATE
    arithmetic.cu
)

target_include_directories(${PROJECT_NAME} PUBLIC "./include")
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 50)
