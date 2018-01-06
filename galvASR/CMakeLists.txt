# This is helpful for diagnosing missing symbols at compile time.
# set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined ${CMAKE_SHARED_LINKER_FLAGS}")

add_subdirectory(python)

if(USE_TENSORFLOW)
  add_subdirectory(tensorflow_ext)
endif()

if(USE_CAFFE2)
  add_subdirectory(caffe2_ext)
endif()
