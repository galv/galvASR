# This is helpful for diagnosing missing symbols at compile time.
# set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined ${CMAKE_SHARED_LINKER_FLAGS}")

if(USE_TENSORFLOW)
  add_subdirectory(tensorflow_ext)
endif()

if(USE_CAFFE2)
  add_subdirectory(caffe2_ext)
endif()

# add_subdirectory(python)

if(PYTHON_INSTALL_EDITABLE)
  ADD_CUSTOM_TARGET(link_python_dir ALL
    COMMAND ${CMAKE_COMMAND} -E create_symlink "${PROJECT_SOURCE_DIR}/galvASR/python"
    "${CMAKE_BINARY_DIR}/galvASR/python")
elseif()
  add_custom_target(python_copy_files ALL)
  file(GLOB_RECURSE PYTHON_SRCS RELATIVE ${PROJECT_SOURCE_DIR}
    "${PROJECT_SOURCE_DIR}/galvASR/python/*.py")
  foreach(python_src ${PYTHON_SRCS})
    get_filename_component(dir ${python_src} DIRECTORY)
    message("${python_src}")
    add_custom_command(
      TARGET python_copy_files PRE_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
      ${PROJECT_SOURCE_DIR}/${python_src} ${CMAKE_BINARY_DIR}/${python_src})
  endforeach()
endif()
