#!/bin/bash

# 确保在脚本执行失败时立即退出
set -e

echo "--- 开始构建和验证卷积算子 ---"

# 最佳实践: 使用单独的 'build' 目录进行树外构建
BUILD_DIR="build"
if [ -d "$BUILD_DIR" ]; then
    echo "清理旧的构建目录..."
    # 使用 sudo rm -rf 确保所有旧文件（即使权限不正确）都能被删除
    sudo rm -rf "$BUILD_DIR"
fi
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# 配置项目: 读取 CMakeLists.txt 并生成 Makefile
echo "1. 配置项目..."
# 使用 cmake .. 来配置两个可执行文件
cmake ..

# 编译两个可执行文件
echo "2. 编译项目..."
# 使用 make 来编译所有目标
make

# 检查编译是否成功
if [ $? -ne 0 ]; then
    echo "编译失败. 退出."
    exit 1
fi

echo "正在生成随机种子文件..."
./seed_generator

echo "--- 开始性能测试 ---"

# 定义日志文件名，使用日期戳确保唯一性，便于后期管理
LOG_FILE="tegrastats_$(date +%Y%m%d_%H%M%S).log"
PERF_TEST_EXE="./convolution_operator"

echo "## 性能测试前准备：启动 tegrastats 功耗监控..."

# 使用 tegrastats 的 --logfile 选项实现可靠日志记录。
# 将其置于后台运行 (&)，以便后续执行性能测试。
# 推荐使用 --interval 100 毫秒，以获得更细粒度的功耗数据。
sudo tegrastats --interval 1000 --logfile "$LOG_FILE" &
TEGRASTATS_PID=$!

echo "## tegrastats 进程已在后台启动,PID: $TEGRASTATS_PID"

# 增加健壮性：在执行测试前，检查 tegrastats 进程是否成功启动。
if ! ps -p $TEGRASTATS_PID > /dev/null
then
    echo "错误: tegrastats 进程未能成功启动。请检查权限或路径。"
    exit 1
fi

echo "## 运行性能测试：$PERF_TEST_EXE"

# 运行用户的性能测试可执行文件
"$PERF_TEST_EXE" > /dev/null

echo "## 性能测试完成。"

# 关键修正：添加2秒的延迟，确保 tegrastats 有足够时间写入数据
echo "## 等待2秒，以便 tegrastats 记录数据..."
sleep 2

# 优雅地停止 tegrastats 进程。
sudo tegrastats --stop

echo "## tegrastats 进程已停止。"
echo "## 功耗数据已记录到文件: $LOG_FILE"
echo "## 你可以使用 'cat $LOG_FILE' 或其他文本编辑器查看数据。"

echo "--- 性能测试完成 ---"

echo "--- 开始结果验证 ---"

echo "3. 运行并保存输出..."

# 运行第一个算子 (im2col + GEMM) 并将输出保存到文件
echo "运行 im2col + GEMM 算子..."
echo "注意：计时信息将直接显示在终端上，不会写入文件。"
./convolution_operator > output_gemm.txt

# 运行第二个算子 (直接卷积) 并将输出保存到文件
echo "运行直接卷积算子..."
./direct_convolution > output_direct.txt

echo "4. 验证结果..."
echo "比较两个算子的输出..."

# 使用新的 C++ 比较工具，而不是 diff
./compare_outputs output_gemm.txt output_direct.txt

if [ $? -eq 0 ]; then
    echo "✅ 验证成功：两个卷积算子的结果在容差范围内一致。"
else
    echo "❌ 验证失败：两个卷积算子的结果不一致。请检查 output_gemm.txt 和 output_direct.txt 文件。"
fi

# 切换回主目录
cd ..

echo "--- 验证完成 ---"
