# This is helpful for diagnosing missing symbols at compile time.
# set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined ${CMAKE_SHARED_LINKER_FLAGS}")

if(USE_TENSORFLOW)
  add_subdirectory(tensorflow_ext)
endif()

if(USE_CAFFE2)
  add_subdirectory(caffe2_ext)
endif()

# add_subdirectory(python)

# Disable this for now. I don't understand yet what the difference
# should be between `python setup.py devel` and
# `CMAKE_ARGS="-DPYTHON_INSTALL_EDIABLE=YES" python setup.py install`
# if(PYTHON_INSTALL_EDITABLE) ADD_CUSTOM_TARGET(link_python_dir ALL
# COMMAND ${CMAKE_COMMAND} -E create_symlink
# "${PROJECT_SOURCE_DIR}/galvASR/python"
# "${CMAKE_BINARY_DIR}/galvASR/python")
# elseif()
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
# endif()

message("python install dir: ${python_site_packages}")
install(DIRECTORY ${CMAKE_BINARY_DIR}/galvASR DESTINATION "${python_site_packages}"
  FILES_MATCHING PATTERN "*.py")
