#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <cmath>
#include <limits>
#include <cctype>

// 检查字符串是否全是数字
bool is_number(const std::string& s) {
    std::string::const_iterator it = s.begin();
    while (it != s.end() && (std::isdigit(*it) || *it == '.' || *it == '+' || *it == '-')) ++it;
    return !s.empty() && it == s.end();
}

// 比较两个浮点数是否在容差范围内相等
bool are_equal(double a, double b, double epsilon = 1e-4) {
    return std::fabs(a - b) <= epsilon * std::max(std::fabs(a), std::fabs(b));
}

int main(int argc, char* argv[]) {
    // 检查参数数量
    if (argc != 3) {
        std::cerr << "用法: " << argv[0] << " <file1> <file2>" << std::endl;
        return 1;
    }

    std::ifstream file1(argv[1]);
    std::ifstream file2(argv[2]);

    if (!file1.is_open() || !file2.is_open()) {
        std::cerr << "错误: 无法打开文件。" << std::endl;
        return 1;
    }

    std::string line1, line2;
    int line_num = 0;
    while (getline(file1, line1) && getline(file2, line2)) {
        line_num++;
        // 忽略包含非数字的行（如“batchsize=”或“输出矩阵过大”的行）
        std::stringstream ss1(line1);
        std::stringstream ss2(line2);
        std::string word1, word2;
        bool numerical_comparison_needed = false;
        
        while (ss1 >> word1 && ss2 >> word2) {
            if (is_number(word1) && is_number(word2)) {
                numerical_comparison_needed = true;
                double val1 = std::stod(word1);
                double val2 = std::stod(word2);
                if (!are_equal(val1, val2)) {
                    std::cerr << "❌ 不一致: 第 " << line_num << " 行" << std::endl;
                    std::cerr << "文件1: " << word1 << std::endl;
                    std::cerr << "文件2: " << word2 << std::endl;
                    return 1; // 失败退出
                }
            } else if (word1 != word2) {
                // 如果不是数字，进行严格的字符串比较
                std::cerr << "❌ 不一致: 第 " << line_num << " 行" << std::endl;
                std::cerr << "文件1: " << word1 << std::endl;
                std::cerr << "文件2: " << word2 << std::endl;
                return 1; // 失败退出
            }
        }
        
        // 检查行尾是否有不一致的额外内容
        if (ss1.rdbuf()->in_avail() != 0 || ss2.rdbuf()->in_avail() != 0) {
            std::cerr << "❌ 不一致: 第 " << line_num << " 行的单词数量不匹配。" << std::endl;
            return 1;
        }
    }

    // 检查文件行数是否匹配
    if (file1.eof() != file2.eof()) {
        std::cerr << "错误: 文件行数不匹配。" << std::endl;
        return 1;
    }

    std::cout << "✅ 验证成功：两个文件内容一致。" << std::endl;
    return 0; // 成功退出
}