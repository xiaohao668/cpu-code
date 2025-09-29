#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cblas.h>
#include <omp.h> // 引入 OpenMP 头文件
#include <algorithm> // 用于 std::random_shuffle 或 std::sample
#include <iomanip> // 用于格式化输出
#include <fstream>

// 生成随机数矩阵
template <typename T>
void generate_random_matrix(std::vector<T>& matrix, int rows, int cols, std::mt19937& gen) {
    std::uniform_real_distribution<> dis(0.0, 1.0);
    matrix.assign(rows * cols, 0.0); // 重新分配并清零
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = dis(gen);
    }
}

// im2col 展开 (并行化版本)
void im2col(const std::vector<double>& input, std::vector<double>& col,
            int batchsize, int ic, int ih, int iw, int kh, int kw) {
    int oh = ih - kh + 1;
    int ow = iw - kw + 1;
    int col_rows = batchsize * oh * ow;
    int col_cols = ic * kh * kw;
    col.resize(col_rows * col_cols);

    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batchsize; ++b) {
        for (int h = 0; h < oh; ++h) {
            for (int w = 0; w < ow; ++w) {
                for (int c_idx = 0; c_idx < ic; ++c_idx) {
                    for (int kh_idx = 0; kh_idx < kh; ++kh_idx) {
                        for (int kw_idx = 0; kw_idx < kw; ++kw_idx) {
                            int input_h = h + kh_idx;
                            int input_w = w + kw_idx;

                            // 计算 im2col 矩阵中的行索引
                            int col_row_idx = ((b * oh + h) * ow + w);

                            // 计算 im2col 矩阵中的列索引
                            int col_col_idx = (((c_idx * kh) + kh_idx) * kw + kw_idx);

                            // 原始输入数据的索引
                            int input_idx = (((b * ic + c_idx) * ih + input_h) * iw + input_w);

                            // 将值赋给 col 矩阵
                            col[col_row_idx * col_cols + col_col_idx] = input[input_idx];
                        }
                    }
                }
            }
        }
    }
}

// 封装 im2col + GEMM 卷积操作
void convolve_im2col_gemm(const std::vector<double>& input, const std::vector<double>& kernel, std::vector<double>& output,
                          int batchsize, int ic, int ih, int iw, int kc, int kh, int kw) {
    int oh = ih - kh + 1;
    int ow = iw - kw + 1;

    // im2col 展开
    std::vector<double> col;
    im2col(input, col, batchsize, ic, ih, iw, kh, kw);

    // 计算矩阵乘法的维度
    int M = batchsize * oh * ow; // A 矩阵的行数
    int N = kc;                 // B 矩阵的列数
    int K = ic * kh * kw;       // A 的列数和 B 的行数

    // 创建一个临时向量来存储 GEMM 的 NHWC 布局输出
    std::vector<double> output_nhwc(M * N);

    // 使用 cblas_dgemm 进行矩阵乘法
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K,
                1.0, col.data(), K,
                kernel.data(), K,
                0.0, output_nhwc.data(), N);
    
    // 将 GEMM 的输出 (NHWC) 转换为 NCHW 布局
    output.resize(batchsize * kc * oh * ow);
    #pragma omp parallel for collapse(4)
    for (int b = 0; b < batchsize; ++b) {
        for (int k_idx = 0; k_idx < kc; ++k_idx) {
            for (int h = 0; h < oh; ++h) {
                for (int w = 0; w < ow; ++w) {
                    // NHWC 索引
                    int nhwc_idx = (((b * oh + h) * ow + w) * kc + k_idx);
                    // NCHW 索引
                    int nchw_idx = (((b * kc + k_idx) * oh + h) * ow + w);
                    output[nchw_idx] = output_nhwc[nhwc_idx];
                }
            }
        }
    }
}

// 打印矩阵的辅助函数
// 适用于 main.cpp 和 direct_convolution.cpp
void print_matrix(const std::vector<double>& matrix, int rows, int cols) {
    if (matrix.empty()) {
        std::cout << std::endl;
        return;
    }

    // 设置高精度和强制固定格式
    std::cout << std::fixed << std::setprecision(8); 
    
    // 打印矩阵的所有元素，并精确控制空格
    for (size_t i = 0; i < matrix.size(); ++i) {
        // 打印当前元素
        std::cout << matrix[i];
        
        // 只有在不是最后一个元素时才打印空格
        if (i < matrix.size() - 1) {
            std::cout << " "; 
        }
    }
    std::cout << std::endl; // 结束这一行
}

int main() {
    // 从文件中读取共享的随机数种子
    unsigned seed = 0;
    std::ifstream seed_file("random_seed.txt");
    if (seed_file.is_open()) {
        seed_file >> seed;
        seed_file.close();
    }
    std::mt19937 gen(seed);
    
    // 定义常见的 batchsize 和 ic 大小
    std::vector<int> common_sizes = {1, 2, 4, 8, 16, 32, 64, 128, 256};

    // 从列表中随机选择 batchsize 和 ic
    std::uniform_int_distribution<> dist_index(0, common_sizes.size() - 1);
    int batchsize = common_sizes[dist_index(gen)];
    int ic = common_sizes[dist_index(gen)];

    std::uniform_int_distribution<> dist_kc(1, 8);
    std::uniform_int_distribution<> dist_hw(8, 64);
    std::uniform_int_distribution<> dist_khw(2, 5);

    int ih = dist_hw(gen);
    int iw = dist_hw(gen);

    int kc = dist_kc(gen);
    int kh = dist_khw(gen);
    int kw = dist_khw(gen);

    // 保证卷积核大小不超过输入大小
    if (kh > ih) kh = ih;
    if (kw > iw) kw = iw;

    int oh = ih - kh + 1;
    int ow = iw - kw + 1;

    std::cout << "batchsize=" << batchsize << ", ic=" << ic
              << ", ih=" << ih << ", iw=" << iw
              << ", kc=" << kc << ", kh=" << kh << ", kw=" << kw
              << " -> oh=" << oh << ", ow=" << ow << std::endl;

    // 随机生成输入和卷积核
    std::vector<double> input(batchsize * ic * ih * iw);
    std::vector<double> kernel(kc * ic * kh * kw);
    generate_random_matrix(input, batchsize * ic * ih * iw, 1, gen);
    generate_random_matrix(kernel, kc * ic * kh * kw, 1, gen);

    // 输出矩阵
    std::vector<double> output;

    // ===== 计时开始 =====
    auto start = std::chrono::high_resolution_clock::now();

    convolve_im2col_gemm(input, kernel, output, batchsize, ic, ih, iw, kc, kh, kw);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    // ===== 计时结束 =====

    // 将计时信息输出到标准错误流，这样就不会被重定向到文件
    std::cerr << "Convolution (im2col + GEMM) execution time: "
              << elapsed.count() << " ms" << std::endl;

     // 始终打印完整的矩阵，以便验证脚本可以工作
    print_matrix(output, 1, output.size());

    return 0;
}
