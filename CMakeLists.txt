cmake_minimum_required(VERSION 3.15...3.30)

# Check if SKBUILD_PROJECT_NAME is defined, otherwise use a default name
if(NOT DEFINED SKBUILD_PROJECT_NAME)
    set(PROJECT_NAME "nifti")
else()
    set(PROJECT_NAME ${SKBUILD_PROJECT_NAME})
endif()

project(${PROJECT_NAME} LANGUAGES C)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags for GNU
if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pedantic -Wall -Wextra -Wno-unused-parameter -Wno-incompatible-pointer-types")
endif()

# Debug/Release flags for non-MSVC compilers
if(NOT MSVC)
  set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g")
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -march=native")
endif()

# Add zlib subdirectory
add_subdirectory(src/zlib)

# Find Python
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Create the target first
FILE(GLOB MyCSources ./src/*.c)
python_add_library(${PROJECT_NAME} MODULE ${MyCSources} WITH_SOABI)

# Try to find NumPy
execute_process(
  COMMAND "${Python_EXECUTABLE}"
  -c "import numpy; print(numpy.get_include())"
  OUTPUT_VARIABLE NumPy_INCLUDE_DIRS
  RESULT_VARIABLE NumPy_NOT_FOUND
  ERROR_QUIET
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NumPy_NOT_FOUND)
  message(WARNING "NumPy not found. Please install it with: pip install numpy")
else()
  # Include NumPy directories
  target_include_directories(${PROJECT_NAME} PUBLIC ${NumPy_INCLUDE_DIRS})
endif()

# Include zlib directories
target_include_directories(${PROJECT_NAME} PRIVATE ${zlib_SOURCE_DIR})

# Include Python directories
target_include_directories(${PROJECT_NAME} PUBLIC ${Python_INCLUDE_DIRS})

# Define feature macro
target_compile_definitions(${PROJECT_NAME} PRIVATE HAVE_ZLIB)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE zlib)

# Installation
install(TARGETS ${PROJECT_NAME} DESTINATION .)