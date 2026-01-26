#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

// ядро cuda для префиксной суммы с использованием разделяемой памяти
__global__ void prefixSum(float* input, float* output, int size) {
    // выделяем разделяемую память для блока (в 2 раза больше для ping-pong буфера)
    extern __shared__ float temp[];
    
    // вычисляем глобальный и локальный индексы
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_idx = threadIdx.x;
    
    // загружаем данные из глобальной памяти в разделяемую
    if (global_idx < size) {
        temp[local_idx] = input[global_idx];
    } else {
        temp[local_idx] = 0.0f;
    }
    
    // синхронизируем потоки чтобы все данные были загружены
    __syncthreads();
    
    // алгоритм параллельного сканирования (up-sweep фаза)
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (local_idx + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            temp[index] += temp[index - stride];
        }
        // синхронизируем после каждого шага
        __syncthreads();
    }
    
    // down-sweep фаза
    for (int stride = blockDim.x / 4; stride > 0; stride /= 2) {
        int index = (local_idx + 1) * stride * 2 - 1;
        if (index + stride < blockDim.x) {
            temp[index + stride] += temp[index];
        }
        // синхронизируем после каждого шага
        __syncthreads();
    }
    
    // записываем результат обратно в глобальную память
    if (global_idx < size) {
        output[global_idx] = temp[local_idx];
    }
}

// cpu версия префиксной суммы для сравнения
void cpuPrefixSum(float* input, float* output, int size) {
    // первый элемент остаётся таким же
    output[0] = input[0];
    // каждый следующий = предыдущий результат + текущий элемент
    for (int i = 1; i < size; i++) {
        output[i] = output[i - 1] + input[i];
    }
}

int main() {
    const int SIZE = 1000000;
    const int BYTES = SIZE * sizeof(float);
    
    // выделяем память на хосте
    float* h_input = new float[SIZE];
    float* h_output_cpu = new float[SIZE];
    float* h_output_gpu = new float[SIZE];
    
    // заполняем массив единицами для простоты проверки
    for (int i = 0; i < SIZE; i++) {
        h_input[i] = 1.0f;
    }
    
    // замеряем время cpu версии
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpuPrefixSum(h_input, h_output_cpu, SIZE);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;
    
    // указатели на память gpu
    float *d_input, *d_output;
    
    // выделяем память на gpu
    cudaMalloc(&d_input, BYTES);
    cudaMalloc(&d_output, BYTES);
    
    // копируем входные данные на gpu
    cudaMemcpy(d_input, h_input, BYTES, cudaMemcpyHostToDevice);
    
    // конфигурация запуска: 256 потоков в блоке
    int threadsPerBlock = 256;
    int blocksPerGrid = (SIZE + threadsPerBlock - 1) / threadsPerBlock;
    // размер разделяемой памяти на блок
    int sharedMemSize = threadsPerBlock * sizeof(float);
    
    // замеряем время gpu версии
    auto gpu_start = std::chrono::high_resolution_clock::now();
    // запускаем ядро с разделяемой памятью
    prefixSum<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_input, d_output, SIZE);
    // ждём завершения
    cudaDeviceSynchronize();
    auto gpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_time = gpu_end - gpu_start;
    
    // копируем результат обратно
    cudaMemcpy(h_output_gpu, d_output, BYTES, cudaMemcpyDeviceToHost);
    
    // проверяем корректность (первые 10 элементов)
    std::cout << "Проверка первых 10 элементов:\n";
    std::cout << "CPU: ";
    for (int i = 0; i < 10; i++) {
        std::cout << h_output_cpu[i] << " ";
    }
    std::cout << "\nGPU: ";
    for (int i = 0; i < 10; i++) {
        std::cout << h_output_gpu[i] << " ";
    }
    std::cout << "\n\n";
    
    // выводим результаты замеров
    std::cout << "Размер массива: " << SIZE << " элементов\n";
    std::cout << "Время CPU: " << cpu_time.count() << " мс\n";
    std::cout << "Время GPU: " << gpu_time.count() << " мс\n";
    std::cout << "Ускорение: " << cpu_time.count() / gpu_time.count() << "x\n";
    
    // освобождаем память
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output_cpu;
    delete[] h_output_gpu;
    
    return 0;
}
