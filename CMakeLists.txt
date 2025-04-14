include_directories(/usr/local/cuda-X.X/include)
link_directories(/usr/local/cuda-X.X/lib64)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(your_project_name ${OpenCV_LIBS})
