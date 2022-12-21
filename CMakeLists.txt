cmake_minimum_required(VERSION 3.14)
project(project)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

add_subdirectory(app)
add_subdirectory(libs)

# lots of warnings and all warnings as errors
# add_compile_options(-Wall -Wextra -pedantic-errors -Werror -Wconversion -Wsign-conversion)