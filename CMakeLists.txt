cmake_minimum_required(VERSION 3.15...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_C_FLAGS "-pedantic -Wall -Wextra -Wno-unused-parameter -Wno-incompatible-pointer-types")
endif(CMAKE_COMPILER_IS_GNUCXX)

set(CMAKE_C_FLAGS_DEBUG "-g")
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native")

add_subdirectory(src/zlib)
include_directories(${zlib_SOURCE_DIR})

# Grab the variables from a local Python installation
# NumPy headers
execute_process(
  COMMAND "${PYTHON_EXECUTABLE}"
  -c "import numpy; print(numpy.get_include())"
  OUTPUT_VARIABLE NumPy_INCLUDE_DIRS
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

FILE(GLOB MyCSources ./src/*.c)
python_add_library(${SKBUILD_PROJECT_NAME} MODULE ${MyCSources} WITH_SOABI)

target_include_directories(${SKBUILD_PROJECT_NAME} PUBLIC
                           ${NumPy_INCLUDE_DIRS}
                           ${PYTHON_INCLUDE_DIRS})

add_compile_definitions(HAVE_ZLIB)

target_link_libraries(${SKBUILD_PROJECT_NAME} PRIVATE zlib)

install(TARGETS ${SKBUILD_PROJECT_NAME} DESTINATION .)
