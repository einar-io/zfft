add_library( ${PROJECT_NAME} SHARED )

target_sources(${PROJECT_NAME}
  PRIVATE
    arithmetic.cu
)


#set_target_properties(${PROJECT_NAME}
#  PROPERTIES OUTPUT_NAME
#zama
#)

#target_include_directories (zama PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${PROJECT_NAME} PUBLIC "../include")
target_include_directories(${PROJECT_NAME} PUBLIC "./include")
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 50)
