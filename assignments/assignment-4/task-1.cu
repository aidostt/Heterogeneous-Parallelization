#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

// ядро cuda для суммирования элементов массива
__global__ void arraySum(float* input, float* output, int size) {
    // вычисляем глобальный индекс потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // каждый поток обрабатывает несколько элементов с шагом равным общему числу потоков
    int stride = blockDim.x * gridDim.x;
    
    // локальная сумма для этого потока
    float sum = 0.0f;
    
    // проходим по массиву с шагом stride
    for (int i = idx; i < size; i += stride) {
        sum += input[i];
    }
    
    // атомарно добавляем локальную сумму к общему результату
    atomicAdd(output, sum);
}

// функция суммирования на cpu для сравнения
float cpuSum(float* arr, int size) {
    float sum = 0.0f;
    // просто проходим по всем элементам и складываем
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum;
}

int main() {
    const int SIZE = 100000;
    const int BYTES = SIZE * sizeof(float);
    
    // выделяем память на хосте (cpu)
    float* h_input = new float[SIZE];
    float h_output = 0.0f;
    
    // заполняем массив случайными числами от 0 до 1
    for (int i = 0; i < SIZE; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // замеряем время cpu версии
    auto cpu_start = std::chrono::high_resolution_clock::now();
    float cpu_result = cpuSum(h_input, SIZE);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;
    
    // указатели на память gpu
    float *d_input, *d_output;
    
    // выделяем память на gpu
    cudaMalloc(&d_input, BYTES);
    cudaMalloc(&d_output, sizeof(float));
    
    // копируем входные данные с cpu на gpu
    cudaMemcpy(d_input, h_input, BYTES, cudaMemcpyHostToDevice);
    // обнуляем выходной результат на gpu
    cudaMemcpy(d_output, &h_output, sizeof(float), cudaMemcpyHostToDevice);
    
    // определяем конфигурацию запуска: 256 потоков в блоке
    int threadsPerBlock = 256;
    // вычисляем количество блоков (округление вверх)
    int blocksPerGrid = (SIZE + threadsPerBlock - 1) / threadsPerBlock;
    
    // замеряем время gpu версии
    auto gpu_start = std::chrono::high_resolution_clock::now();
    // запускаем ядро на gpu
    arraySum<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, SIZE);
    // ждём завершения всех операций на gpu
    cudaDeviceSynchronize();
    auto gpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_time = gpu_end - gpu_start;
    
    // копируем результат обратно с gpu на cpu
    float gpu_result;
    cudaMemcpy(&gpu_result, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    
    // выводим результаты
    std::cout << "Размер массива: " << SIZE << " элементов\n";
    std::cout << "CPU сумма: " << cpu_result << "\n";
    std::cout << "GPU сумма: " << gpu_result << "\n";
    std::cout << "Время CPU: " << cpu_time.count() << " мс\n";
    std::cout << "Время GPU: " << gpu_time.count() << " мс\n";
    std::cout << "Ускорение: " << cpu_time.count() / gpu_time.count() << "x\n";
    
    // освобождаем память gpu
    cudaFree(d_input);
    cudaFree(d_output);
    
    // освобождаем память cpu
    delete[] h_input;
    
    return 0;
}
