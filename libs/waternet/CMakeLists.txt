cmake_minimum_required(VERSION 3.14)
project(waternet)

set(CMAKE_PREFIX_PATH /home/user_name/libtorch_cpu)
set(CMAKE_INSTALL_PREFIX /usr/local)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

file(GLOB_RECURSE SOURCES "src/*.cpp" "inc/*.hpp")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(${PROJECT_NAME} STATIC ${SOURCES})
add_library(libs::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME} PUBLIC "${OpenCV_LIBS}" "${TORCH_LIBRARIES}")

target_include_directories(${PROJECT_NAME}
        PUBLIC
        $<INSTALL_INTERFACE:inc>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
        ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
