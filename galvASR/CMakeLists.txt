# This is helpful for diagnosing missing symbols at compile time.
# set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined ${CMAKE_SHARED_LINKER_FLAGS}")

if(USE_TENSORFLOW)
  add_subdirectory(tensorflow_ext)
endif()

# Put this into the source directory to make an "editable" install
# easier later.
configure_file(${PROJECT_SOURCE_DIR}/galvASR/python/constants.py.in
  ${PROJECT_SOURCE_DIR}/galvASR/python/constants.py)
