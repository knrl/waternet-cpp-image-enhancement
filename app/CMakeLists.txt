cmake_minimum_required(VERSION 3.14)
project(App)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

file(GLOB_RECURSE SOURCES "src/*.cpp" )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_executable(${PROJECT_NAME} ${SOURCES} )
target_link_libraries(${PROJECT_NAME} PUBLIC
        libs::waternet
        )