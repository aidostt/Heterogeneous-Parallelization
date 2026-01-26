#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <thread>

// простое ядро cuda для обработки массива (умножение на 2)
__global__ void processArrayGPU(float* input, float* output, int size, int offset) {
    // вычисляем глобальный индекс с учётом смещения
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // проверяем границы
    if (idx < size) {
        // умножаем элемент на 2
        output[idx] = input[idx] * 2.0f;
    }
}

// функция обработки массива на cpu (умножение на 2)
void processArrayCPU(float* input, float* output, int size) {
    // просто проходим по массиву и умножаем каждый элемент на 2
    for (int i = 0; i < size; i++) {
        output[i] = input[i] * 2.0f;
    }
}

// функция обработки только на cpu для сравнения
void cpuOnly(float* input, float* output, int size) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // обрабатываем весь массив на cpu
    processArrayCPU(input, output, size);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> time = end - start;
    
    std::cout << "Только CPU: " << time.count() << " мс\n";
}

// функция обработки только на gpu для сравнения
void gpuOnly(float* h_input, float* h_output, int size) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // выделяем память на gpu
    float *d_input, *d_output;
    int bytes = size * sizeof(float);
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    
    // копируем данные на gpu
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    // конфигурация запуска
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    // запускаем ядро
    processArrayGPU<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, size, 0);
    
    // ждём завершения
    cudaDeviceSynchronize();
    
    // копируем результат обратно
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    
    // освобождаем память
    cudaFree(d_input);
    cudaFree(d_output);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> time = end - start;
    
    std::cout << "Только GPU: " << time.count() << " мс\n";
}

// гибридная функция: cpu и gpu работают параллельно
void hybridProcessing(float* h_input, float* h_output, int size) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // делим массив пополам
    int half_size = size / 2;
    int bytes_half = half_size * sizeof(float);
    int remaining_size = size - half_size;
    int bytes_remaining = remaining_size * sizeof(float);
    
    // указатели на половины массива
    float* cpu_input = h_input;
    float* cpu_output = h_output;
    float* gpu_input = h_input + half_size;
    float* gpu_output = h_output + half_size;
    
    // выделяем память на gpu для второй половины
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes_remaining);
    cudaMalloc(&d_output, bytes_remaining);
    
    // копируем вторую половину на gpu
    cudaMemcpy(d_input, gpu_input, bytes_remaining, cudaMemcpyHostToDevice);
    
    // конфигурация запуска для gpu
    int threadsPerBlock = 256;
    int blocksPerGrid = (remaining_size + threadsPerBlock - 1) / threadsPerBlock;
    
    // запускаем обработку на gpu (асинхронно)
    processArrayGPU<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, remaining_size, 0);
    
    // одновременно запускаем обработку первой половины на cpu в отдельном потоке
    std::thread cpu_thread([cpu_input, cpu_output, half_size]() {
        processArrayCPU(cpu_input, cpu_output, half_size);
    });
    
    // ждём завершения gpu
    cudaDeviceSynchronize();
    
    // копируем результат gpu обратно
    cudaMemcpy(gpu_output, d_output, bytes_remaining, cudaMemcpyDeviceToHost);
    
    // ждём завершения cpu потока
    cpu_thread.join();
    
    // освобождаем память gpu
    cudaFree(d_input);
    cudaFree(d_output);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> time = end - start;
    
    std::cout << "Гибрид (CPU+GPU): " << time.count() << " мс\n";
}

int main() {
    const int SIZE = 10000000; // 10 миллионов элементов
    const int BYTES = SIZE * sizeof(float);
    
    // выделяем память на хосте
    float* h_input = new float[SIZE];
    float* h_output = new float[SIZE];
    
    // заполняем входной массив
    for (int i = 0; i < SIZE; i++) {
        h_input[i] = static_cast<float>(i);
    }
    
    std::cout << "Размер массива: " << SIZE << " элементов\n";
    std::cout << "Задача: умножение каждого элемента на 2\n\n";
    
    // тестируем только cpu
    cpuOnly(h_input, h_output, SIZE);
    
    // тестируем только gpu
    gpuOnly(h_input, h_output, SIZE);
    
    // тестируем гибридную версию
    hybridProcessing(h_input, h_output, SIZE);
    
    // проверяем корректность результата (первые 5 элементов)
    std::cout << "\nПроверка результата (первые 5 элементов):\n";
    std::cout << "Вход: ";
    for (int i = 0; i < 5; i++) {
        std::cout << h_input[i] << " ";
    }
    std::cout << "\nВыход: ";
    for (int i = 0; i < 5; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << "\n";
    
    // освобождаем память
    delete[] h_input;
    delete[] h_output;
    
    return 0;
}
