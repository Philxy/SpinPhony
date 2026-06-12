#include <iostream>
#include "cpu_logic.h"
#include "gpu_logic.cuh"

int main() {
    std::cout << "--- Starting SpinPhony ---" << std::endl;
    
    // Execute CPU logic
    initialize_cpu_lattice();
    
    // Execute GPU logic
    launch_cuda_test();
    
    std::cout << "--- Simulation Complete ---" << std::endl;
    return 0;
}