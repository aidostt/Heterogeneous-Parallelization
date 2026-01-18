#include <iostream>
#include <cuda_runtime.h>

// ядро cuda - функция, которая выполняется на gpu
// __global__ означает, что эта функция вызывается с cpu, но выполняется на gpu
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    // получаем уникальный индекс текущего потока
    // blockIdx.x - номер блока, blockDim.x - количество потоков в блоке
    // threadIdx.x - номер потока внутри блока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // проверяем, что индекс не выходит за границы массива
    // это важно, потому что количество потоков может быть больше размера массива
    if (idx < n) {
        // складываем элементы массивов a и b, результат записываем в c
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    // размер массивов
    const int n = 1000000;
    // размер в байтах
    const size_t bytes = n * sizeof(float);
    
    // выделяем память на хосте (cpu)
    float* h_a = new float[n]; // первый входной массив
    float* h_b = new float[n]; // второй входной массив
    float* h_c = new float[n]; // массив для результата
    
    // инициализируем входные массивы
    for (int i = 0; i < n; i++) {
        h_a[i] = static_cast<float>(i); // заполняем числами 0, 1, 2, ...
        h_b[i] = static_cast<float>(i * 2); // заполняем числами 0, 2, 4, ...
    }
    
    // указатели на память gpu (device)
    float* d_a;
    float* d_b;
    float* d_c;
    
    // выделяем память на gpu
    cudaMalloc(&d_a, bytes); // для первого массива
    cudaMalloc(&d_b, bytes); // для второго массива
    cudaMalloc(&d_c, bytes); // для результата
    
    // копируем данные с cpu на gpu
    // cudaMemcpyHostToDevice означает копирование с хоста на устройство
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    // настраиваем параметры запуска ядра
    int threadsPerBlock = 256; // количество потоков в одном блоке
    // вычисляем количество блоков, округляя вверх
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // запускаем ядро на gpu
    // <<<blocksPerGrid, threadsPerBlock>>> - синтаксис cuda для указания конфигурации запуска
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    
    // ждем завершения всех операций на gpu
    cudaDeviceSynchronize();
    
    // копируем результат обратно с gpu на cpu
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    // проверяем корректность результата (первые 10 элементов)
    std::cout << "проверка первых 10 элементов:\n";
    for (int i = 0; i < 10; i++) {
        // ожидаемое значение: i + i*2 = i*3
        std::cout << h_a[i] << " + " << h_b[i] << " = " << h_c[i];
        std::cout << " (ожидается: " << (h_a[i] + h_b[i]) << ")\n";
    }
    
    // освобождаем память на gpu
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    // освобождаем память на cpu
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    
    std::cout << "\nпрограмма успешно завершена\n";
    return 0;
}
