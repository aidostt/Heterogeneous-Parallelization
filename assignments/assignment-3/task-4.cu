#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

// простое ядро редукции (неэффективное, для сравнения)
__global__ void reductionSimple(const float* input, float* output, int n) {
    // используем только первый поток для суммирования
    // это очень медленно, но показывает базовую идею
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += input[i];
        }
        output[0] = sum;
    }
}

// оптимизированное ядро редукции с shared memory
__global__ void reductionShared(const float* input, float* output, int n) {
    // выделяем shared memory для частичных сумм блока
    extern __shared__ float sdata[];
    
    // индекс потока
    unsigned int tid = threadIdx.x;
    // глобальный индекс элемента
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // каждый поток загружает один элемент в shared memory
    // если индекс выходит за границы, загружаем 0
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads(); // ждем, пока все потоки загрузят данные
    
    // выполняем редукцию в shared memory
    // на каждой итерации количество активных потоков уменьшается вдвое
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            // каждый активный поток суммирует свой элемент с элементом на расстоянии s
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads(); // синхронизируем перед следующей итерацией
    }
    
    // первый поток блока записывает результат в глобальную память
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// улучшенное ядро редукции с меньшим количеством конфликтов банков памяти
__global__ void reductionOptimized(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // каждый поток загружает два элемента и сразу их суммирует
    // это уменьшает количество данных вдвое с самого начала
    float sum = 0.0f;
    if (idx < n) sum += input[idx];
    if (idx + blockDim.x < n) sum += input[idx + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();
    
    // выполняем редукцию
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // последние 32 потока (один warp) работают без синхронизации
    // потому что warp выполняется синхронно на gpu
    if (tid < 32) {
        // эти операции выполняются внутри одного warp, синхронизация не нужна
        volatile float* smem = sdata; // volatile для предотвращения оптимизаций компилятора
        if (blockDim.x >= 64) smem[tid] += smem[tid + 32];
        if (blockDim.x >= 32) smem[tid] += smem[tid + 16];
        if (blockDim.x >= 16) smem[tid] += smem[tid + 8];
        if (blockDim.x >= 8) smem[tid] += smem[tid + 4];
        if (blockDim.x >= 4) smem[tid] += smem[tid + 2];
        if (blockDim.x >= 2) smem[tid] += smem[tid + 1];
    }
    
    // записываем результат блока
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// ядро для поиска максимума (вариация редукции)
__global__ void reductionMax(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // загружаем данные
    sdata[tid] = (idx < n) ? input[idx] : -INFINITY; // для максимума используем -бесконечность
    __syncthreads();
    
    // выполняем редукцию с операцией max вместо суммы
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            // находим максимум из двух элементов
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// функция для вычисления суммы на cpu (для проверки)
float sumCPU(const float* data, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum;
}

// функция для нахождения максимума на cpu
float maxCPU(const float* data, int n) {
    float maxVal = data[0];
    for (int i = 1; i < n; i++) {
        if (data[i] > maxVal) {
            maxVal = data[i];
        }
    }
    return maxVal;
}

int main() {
    // размер массива
    const int n = 1 << 24; // 16 миллионов элементов
    const size_t bytes = n * sizeof(float);
    
    // выделяем память на хосте
    float* h_input = new float[n];
    
    // инициализируем массив случайными числами
    std::cout << "инициализация массива из " << n << " элементов...\n";
    for (int i = 0; i < n; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX; // числа от 0 до 1
    }
    
    // вычисляем правильный ответ на cpu
    std::cout << "вычисление суммы на cpu...\n";
    float cpuSum = sumCPU(h_input, n);
    float cpuMax = maxCPU(h_input, n);
    std::cout << "сумма на cpu: " << cpuSum << "\n";
    std::cout << "максимум на cpu: " << cpuMax << "\n";
    
    // выделяем память на gpu
    float* d_input;
    float* d_temp; // для промежуточных результатов
    float* d_output; // для финального результата
    
    cudaMalloc(&d_input, bytes);
    
    // копируем данные на gpu
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    // настраиваем конфигурацию запуска
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // выделяем память для результатов каждого блока
    cudaMalloc(&d_temp, blocksPerGrid * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));
    
    // создаем события для измерения времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;
    
    // тест простого ядра (только для маленького массива, иначе очень долго)
    if (n <= 1000000) {
        std::cout << "\nзапуск простого ядра...\n";
        cudaEventRecord(start);
        reductionSimple<<<1, 1>>>(d_input, d_output, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "время простого ядра: " << milliseconds << " мс\n";
    }
    
    // тест ядра с shared memory
    std::cout << "\nзапуск ядра с shared memory...\n";
    cudaEventRecord(start);
    
    // первый проход - редукция блоков
    reductionShared<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        d_input, d_temp, n);
    
    // второй проход - редукция результатов блоков
    int finalBlocks = (blocksPerGrid + threadsPerBlock - 1) / threadsPerBlock;
    reductionShared<<<finalBlocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        d_temp, d_output, blocksPerGrid);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "время ядра с shared memory: " << milliseconds << " мс\n";
    
    // копируем результат
    float gpuSum;
    cudaMemcpy(&gpuSum, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "сумма на gpu: " << gpuSum << "\n";
    std::cout << "ошибка: " << abs(gpuSum - cpuSum) << "\n";
    
    // тест оптимизированного ядра
    std::cout << "\nзапуск оптимизированного ядра...\n";
    int optBlocks = (n + (threadsPerBlock * 2) - 1) / (threadsPerBlock * 2);
    
    cudaEventRecord(start);
    
    // первый проход
    reductionOptimized<<<optBlocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        d_input, d_temp, n);
    
    // второй проход
    int finalOptBlocks = (optBlocks + (threadsPerBlock * 2) - 1) / (threadsPerBlock * 2);
    reductionOptimized<<<finalOptBlocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        d_temp, d_output, optBlocks);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "время оптимизированного ядра: " << milliseconds << " мс\n";
    
    cudaMemcpy(&gpuSum, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "сумма на gpu: " << gpuSum << "\n";
    std::cout << "ошибка: " << abs(gpuSum - cpuSum) << "\n";
    
    // тест поиска максимума
    std::cout << "\nзапуск ядра поиска максимума...\n";
    cudaEventRecord(start);
    
    // первый проход
    reductionMax<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        d_input, d_temp, n);
    
    // второй проход
    reductionMax<<<1, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        d_temp, d_output, blocksPerGrid);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "время поиска максимума: " << milliseconds << " мс\n";
    
    float gpuMax;
    cudaMemcpy(&gpuMax, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "максимум на gpu: " << gpuMax << "\n";
    std::cout << "ошибка: " << abs(gpuMax - cpuMax) << "\n";
    
    // освобождаем память
    cudaFree(d_input);
    cudaFree(d_temp);
    cudaFree(d_output);
    
    delete[] h_input;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    std::cout << "\nпрограмма завершена\n";
    return 0;
}
