# 设置 CMake 的最低版本要求
cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(ConvolutionOperators)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)

# 查找并配置 OpenMP
find_package(OpenMP REQUIRED)

# 手动指定 OpenBLAS 头文件和库路径
set(OpenBLAS_INCLUDE_DIR "/usr/include/aarch64-linux-gnu")
set(OpenBLAS_LIBRARIES "/usr/lib/aarch64-linux-gnu/libopenblas.so")

# 设置 OpenBLAS 的头文件路径
include_directories(${OpenBLAS_INCLUDE_DIR})

# 创建第一个可执行文件: convolution_operator
add_executable(convolution_operator main.cpp)

# 链接 OpenBLAS 和 OpenMP 库
# OpenBLAS 库只用于 convolution_operator
target_link_libraries(convolution_operator ${OpenBLAS_LIBRARIES} OpenMP::OpenMP_CXX)

# 添加 OpenMP 编译选项
target_compile_options(convolution_operator PRIVATE ${OpenMP_CXX_FLAGS})

# 创建第二个可执行文件: direct_convolution
add_executable(direct_convolution direct_convolution.cpp)

# 链接 OpenMP 库
# direct_convolution 只使用 OpenMP，不使用 OpenBLAS
target_link_libraries(direct_convolution OpenMP::OpenMP_CXX)

# 添加 OpenMP 编译选项
target_compile_options(direct_convolution PRIVATE ${OpenMP_CXX_FLAGS})

# 新增可执行文件：用于比较两个算子的输出
add_executable(compare_outputs compare_outputs.cpp)

# 新增：链接 OpenMP
target_link_libraries(compare_outputs OpenMP::OpenMP_CXX)

# 新增可执行文件：用于生成随机种子
add_executable(seed_generator seed_generator.cpp)
