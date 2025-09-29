#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
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

// 直接卷积实现
void direct_convolve(const std::vector<double>& input, const std::vector<double>& kernel, std::vector<double>& output,
                     int batchsize, int ic, int ih, int iw, int kc, int kh, int kw) {
    int oh = ih - kh + 1;
    int ow = iw - kw + 1;
    
    // 调整输出矩阵大小
    output.resize(batchsize * kc * oh * ow, 0);

    // 核心卷积计算循环
    for (int b = 0; b < batchsize; ++b) {
        for (int k_idx = 0; k_idx < kc; ++k_idx) { // 输出通道 (kernel count)
            for (int h = 0; h < oh; ++h) {
                for (int w = 0; w < ow; ++w) {
                    double sum = 0.0;
                    for (int c_idx = 0; c_idx < ic; ++c_idx) { // 输入通道 (input channel)
                        for (int kh_idx = 0; kh_idx < kh; ++kh_idx) {
                            for (int kw_idx = 0; kw_idx < kw; ++kw_idx) {
                                // 计算输入和卷积核的索引
                                int input_h_idx = h + kh_idx;
                                int input_w_idx = w + kw_idx;

                                int input_idx = ((b * ic + c_idx) * ih + input_h_idx) * iw + input_w_idx;
                                // im2col展开后的列索引
                                // 展平为 [ic*kh*kw]
                                int col_K_idx = (((c_idx * kh) + kh_idx) * kw + kw_idx);
                                // im2col展开后的行索引，对应 dgemm 的 A 矩阵列索引
                                // 展平为 [kc]
                                int kernel_M_idx = k_idx;

                                // kernel 矩阵的索引，注意其在 im2col + GEMM 中被视为一个 [kc] x [ic*kh*kw] 的矩阵
                                int kernel_idx = kernel_M_idx * (ic * kh * kw) + col_K_idx;
                                
                                sum += input[input_idx] * kernel[kernel_idx];
                            }
                        }
                    }
                    // 计算输出矩阵的索引
                    int output_idx = ((b * kc + k_idx) * oh + h) * ow + w;
                    output[output_idx] = sum;
                }
            }
        }
    }
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

    // 进行直接卷积计算
    direct_convolve(input, kernel, output, batchsize, ic, ih, iw, kc, kh, kw);

     // 始终打印完整的矩阵，以便验证脚本可以工作
    print_matrix(output, 1, output.size());

    return 0;
}
