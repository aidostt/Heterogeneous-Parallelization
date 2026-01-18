#include <iostream>
#include <cuda_runtime.h>

// размер тайла для shared memory оптимизации
#define TILE_SIZE 16

// простое ядро умножения матриц без оптимизации
__global__ void matrixMulSimple(const float* a, const float* b, float* c, int width) {
    // вычисляем строку и столбец элемента результата
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // проверяем границы матрицы
    if (row < width && col < width) {
        float sum = 0.0f; // накопитель для суммы произведений
        
        // вычисляем скалярное произведение строки на столбец
        for (int k = 0; k < width; k++) {
            // a[row][k] * b[k][col]
            sum += a[row * width + k] * b[k * width + col];
        }
        
        // записываем результат в c[row][col]
        c[row * width + col] = sum;
    }
}

// оптимизированное ядро с использованием shared memory
__global__ void matrixMulTiled(const float* a, const float* b, float* c, int width) {
    // выделяем shared memory для тайлов матриц a и b
    // __shared__ означает, что память разделяется между всеми потоками в блоке
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    // координаты элемента в результирующей матрице
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f; // накопитель для результата
    
    // количество тайлов, которые нужно обработать
    int numTiles = (width + TILE_SIZE - 1) / TILE_SIZE;
    
    // проходим по всем тайлам
    for (int t = 0; t < numTiles; t++) {
        // загружаем тайл из матрицы a в shared memory
        // проверяем границы, чтобы не выйти за пределы матрицы
        if (row < width && (t * TILE_SIZE + threadIdx.x) < width) {
            tileA[threadIdx.y][threadIdx.x] = a[row * width + t * TILE_SIZE + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f; // заполняем нулями, если вышли за границы
        }
        
        // загружаем тайл из матрицы b в shared memory
        if ((t * TILE_SIZE + threadIdx.y) < width && col < width) {
            tileB[threadIdx.y][threadIdx.x] = b[(t * TILE_SIZE + threadIdx.y) * width + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // синхронизируем потоки блока, чтобы все закончили загрузку
        // это важно, потому что все потоки должны видеть полностью загруженные тайлы
        __syncthreads();
        
        // вычисляем частичное произведение для текущего тайла
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        
        // снова синхронизируемся перед загрузкой следующего тайла
        // это нужно, чтобы никто не начал перезаписывать тайл, пока другие его используют
        __syncthreads();
    }
    
    // записываем результат в глобальную память
    if (row < width && col < width) {
        c[row * width + col] = sum;
    }
}

// функция для проверки корректности результата на cpu
void matrixMulCPU(const float* a, const float* b, float* c, int width) {
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            float sum = 0.0f;
            for (int k = 0; k < width; k++) {
                sum += a[row * width + k] * b[k * width + col];
            }
            c[row * width + col] = sum;
        }
    }
}

int main() {
    // размер квадратной матрицы
    const int width = 1024;
    const size_t bytes = width * width * sizeof(float);
    
    // выделяем память на хосте
    float* h_a = new float[width * width]; // первая матрица
    float* h_b = new float[width * width]; // вторая матрица
    float* h_c = new float[width * width]; // результат
    
    // инициализируем матрицы
    for (int i = 0; i < width * width; i++) {
        h_a[i] = static_cast<float>(rand()) / RAND_MAX; // случайные числа от 0 до 1
        h_b[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // указатели на память gpu
    float* d_a;
    float* d_b;
    float* d_c;
    
    // выделяем память на gpu
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    // копируем данные на gpu
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    // настраиваем конфигурацию запуска
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE); // блок 16x16 потоков
    // вычисляем количество блоков в каждом измерении
    dim3 blocksPerGrid((width + TILE_SIZE - 1) / TILE_SIZE, 
                       (width + TILE_SIZE - 1) / TILE_SIZE);
    
    // создаем события для измерения времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // запускаем простое ядро и измеряем время
    std::cout << "запуск простого ядра...\n";
    cudaEventRecord(start);
    matrixMulSimple<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "время простого ядра: " << milliseconds << " мс\n";
    
    // запускаем оптимизированное ядро и измеряем время
    std::cout << "запуск оптимизированного ядра...\n";
    cudaEventRecord(start);
    matrixMulTiled<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "время оптимизированного ядра: " << milliseconds << " мс\n";
    
    // копируем результат обратно
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    // проверяем корректность на маленькой матрице (для экономии времени)
    std::cout << "\nпроверка корректности на матрице 4x4...\n";
    const int testWidth = 4;
    float* test_a = new float[testWidth * testWidth];
    float* test_b = new float[testWidth * testWidth];
    float* test_c_gpu = new float[testWidth * testWidth];
    float* test_c_cpu = new float[testWidth * testWidth];
    
    // инициализируем тестовые матрицы
    for (int i = 0; i < testWidth * testWidth; i++) {
        test_a[i] = static_cast<float>(i);
        test_b[i] = static_cast<float>(i);
    }
    
    // вычисляем на cpu
    matrixMulCPU(test_a, test_b, test_c_cpu, testWidth);
    
    // вычисляем на gpu
    float *d_test_a, *d_test_b, *d_test_c;
    size_t testBytes = testWidth * testWidth * sizeof(float);
    cudaMalloc(&d_test_a, testBytes);
    cudaMalloc(&d_test_b, testBytes);
    cudaMalloc(&d_test_c, testBytes);
    
    cudaMemcpy(d_test_a, test_a, testBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_test_b, test_b, testBytes, cudaMemcpyHostToDevice);
    
    dim3 testBlocks(1, 1);
    dim3 testThreads(testWidth, testWidth);
    matrixMulSimple<<<testBlocks, testThreads>>>(d_test_a, d_test_b, d_test_c, testWidth);
    
    cudaMemcpy(test_c_gpu, d_test_c, testBytes, cudaMemcpyDeviceToHost);
    
    // сравниваем результаты
    bool correct = true;
    for (int i = 0; i < testWidth * testWidth; i++) {
        if (abs(test_c_gpu[i] - test_c_cpu[i]) > 1e-3) {
            correct = false;
            break;
        }
    }
    
    if (correct) {
        std::cout << "результаты совпадают! gpu работает корректно.\n";
    } else {
        std::cout << "ошибка: результаты не совпадают!\n";
    }
    
    // освобождаем память
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_test_a);
    cudaFree(d_test_b);
    cudaFree(d_test_c);
    
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    delete[] test_a;
    delete[] test_b;
    delete[] test_c_gpu;
    delete[] test_c_cpu;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    std::cout << "\nпрограмма завершена\n";
    return 0;
}
