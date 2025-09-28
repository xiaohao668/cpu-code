#include <iostream>
#include <fstream>
#include <chrono>

int main() {
    unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::ofstream seed_file("random_seed.txt");
    if (seed_file.is_open()) {
        seed_file << seed;
        seed_file.close();
    }
    return 0;
}