#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

// размер ядра свёртки (фильтра)
#define KERNEL_SIZE 5
// радиус ядра (сколько пикселей в каждую сторону от центра)
#define KERNEL_RADIUS (KERNEL_SIZE / 2)

// простое ядро свёртки без оптимизации
__global__ void convolutionSimple(const float* input, float* output, 
                                   const float* kernel, int width, int height) {
    // вычисляем координаты пикселя, который обрабатывает этот поток
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // проверяем, что координаты внутри изображения
    if (row < height && col < width) {
        float sum = 0.0f; // накопитель для результата свёртки
        
        // проходим по всем элементам ядра свёртки
        for (int ky = -KERNEL_RADIUS; ky <= KERNEL_RADIUS; ky++) {
            for (int kx = -KERNEL_RADIUS; kx <= KERNEL_RADIUS; kx++) {
                // вычисляем координаты соседнего пикселя
                int imageRow = row + ky;
                int imageCol = col + kx;
                
                // проверяем границы изображения
                if (imageRow >= 0 && imageRow < height && 
                    imageCol >= 0 && imageCol < width) {
                    // индекс в массиве изображения
                    int imageIdx = imageRow * width + imageCol;
                    // индекс в массиве ядра (смещаем, чтобы индексы были >= 0)
                    int kernelIdx = (ky + KERNEL_RADIUS) * KERNEL_SIZE + (kx + KERNEL_RADIUS);
                    // умножаем значение пикселя на соответствующий вес ядра
                    sum += input[imageIdx] * kernel[kernelIdx];
                }
            }
        }
        
        // записываем результат в выходное изображение
        output[row * width + col] = sum;
    }
}

// оптимизированное ядро с использованием shared memory
__global__ void convolutionShared(const float* input, float* output,
                                   const float* kernel, int width, int height) {
    // размер блока потоков
    const int blockWidth = blockDim.x;
    const int blockHeight = blockDim.y;
    
    // размер тайла в shared memory (блок + границы для свёртки)
    const int tileWidth = blockWidth + 2 * KERNEL_RADIUS;
    const int tileHeight = blockHeight + 2 * KERNEL_RADIUS;
    
    // выделяем shared memory для тайла изображения
    extern __shared__ float tile[];
    
    // глобальные координаты пикселя
    int col = blockIdx.x * blockWidth + threadIdx.x;
    int row = blockIdx.y * blockHeight + threadIdx.y;
    
    // координаты левого верхнего угла тайла в глобальной памяти
    int tileStartCol = blockIdx.x * blockWidth - KERNEL_RADIUS;
    int tileStartRow = blockIdx.y * blockHeight - KERNEL_RADIUS;
    
    // каждый поток загружает несколько элементов в shared memory
    // это нужно, потому что тайл больше, чем количество потоков в блоке
    for (int ty = threadIdx.y; ty < tileHeight; ty += blockHeight) {
        for (int tx = threadIdx.x; tx < tileWidth; tx += blockWidth) {
            // глобальные координаты для загрузки
            int loadRow = tileStartRow + ty;
            int loadCol = tileStartCol + tx;
            
            // индекс в shared memory
            int tileIdx = ty * tileWidth + tx;
            
            // проверяем границы и загружаем значение
            if (loadRow >= 0 && loadRow < height && 
                loadCol >= 0 && loadCol < width) {
                tile[tileIdx] = input[loadRow * width + loadCol];
            } else {
                // за границами изображения используем 0
                tile[tileIdx] = 0.0f;
            }
        }
    }
    
    // синхронизируем потоки, чтобы весь тайл был загружен
    __syncthreads();
    
    // теперь вычисляем свёртку, используя данные из shared memory
    if (row < height && col < width) {
        float sum = 0.0f;
        
        // проходим по ядру свёртки
        for (int ky = -KERNEL_RADIUS; ky <= KERNEL_RADIUS; ky++) {
            for (int kx = -KERNEL_RADIUS; kx <= KERNEL_RADIUS; kx++) {
                // координаты в тайле (смещены на KERNEL_RADIUS от позиции потока)
                int tileRow = threadIdx.y + KERNEL_RADIUS + ky;
                int tileCol = threadIdx.x + KERNEL_RADIUS + kx;
                int tileIdx = tileRow * tileWidth + tileCol;
                
                // индекс в ядре
                int kernelIdx = (ky + KERNEL_RADIUS) * KERNEL_SIZE + (kx + KERNEL_RADIUS);
                
                // умножаем и накапливаем
                sum += tile[tileIdx] * kernel[kernelIdx];
            }
        }
        
        // записываем результат
        output[row * width + col] = sum;
    }
}

// функция для создания гауссова ядра размытия
void createGaussianKernel(float* kernel, int size, float sigma) {
    float sum = 0.0f;
    int radius = size / 2;
    
    // вычисляем значения гауссовой функции
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            int idx = (y + radius) * size + (x + radius);
            // формула гауссовой функции
            float value = exp(-(x*x + y*y) / (2 * sigma * sigma));
            kernel[idx] = value;
            sum += value;
        }
    }
    
    // нормализуем ядро, чтобы сумма весов была равна 1
    for (int i = 0; i < size * size; i++) {
        kernel[i] /= sum;
    }
}

// функция для создания ядра выделения границ (sobel)
void createSobelKernel(float* kernel) {
    // горизонтальный оператор собеля (для демонстрации используем упрощенный 5x5)
    float sobelX[KERNEL_SIZE * KERNEL_SIZE] = {
        -1, -2, 0, 2, 1,
        -2, -3, 0, 3, 2,
        -3, -5, 0, 5, 3,
        -2, -3, 0, 3, 2,
        -1, -2, 0, 2, 1
    };
    
    for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; i++) {
        kernel[i] = sobelX[i];
    }
}

int main() {
    // размеры изображения
    const int width = 1920;
    const int height = 1080;
    const size_t imageBytes = width * height * sizeof(float);
    const size_t kernelBytes = KERNEL_SIZE * KERNEL_SIZE * sizeof(float);
    
    // выделяем память на хосте
    float* h_input = new float[width * height]; // входное изображение
    float* h_output = new float[width * height]; // выходное изображение
    float* h_kernel = new float[KERNEL_SIZE * KERNEL_SIZE]; // ядро свёртки
    
    // создаем тестовое изображение (градиент с шумом)
    std::cout << "создание тестового изображения...\n";
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            // создаем градиент + немного шума
            h_input[idx] = (float)x / width + ((rand() % 100) / 1000.0f);
        }
    }
    
    // создаем гауссово ядро для размытия
    std::cout << "создание ядра свёртки (гауссово размытие)...\n";
    createGaussianKernel(h_kernel, KERNEL_SIZE, 1.0f);
    
    // можно раскомментировать для теста с оператором собеля
    // std::cout << "создание ядра свёртки (sobel)...\n";
    // createSobelKernel(h_kernel);
    
    // выводим ядро для проверки
    std::cout << "ядро свёртки:\n";
    for (int y = 0; y < KERNEL_SIZE; y++) {
        for (int x = 0; x < KERNEL_SIZE; x++) {
            std::cout << h_kernel[y * KERNEL_SIZE + x] << " ";
        }
        std::cout << "\n";
    }
    
    // указатели на память gpu
    float* d_input;
    float* d_output;
    float* d_kernel;
    
    // выделяем память на gpu
    cudaMalloc(&d_input, imageBytes);
    cudaMalloc(&d_output, imageBytes);
    cudaMalloc(&d_kernel, kernelBytes);
    
    // копируем данные на gpu
    cudaMemcpy(d_input, h_input, imageBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernelBytes, cudaMemcpyHostToDevice);
    
    // настраиваем конфигурацию запуска
    dim3 threadsPerBlock(16, 16); // блок 16x16 потоков
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    // размер shared memory для оптимизированной версии
    int sharedMemSize = (threadsPerBlock.x + 2 * KERNEL_RADIUS) * 
                        (threadsPerBlock.y + 2 * KERNEL_RADIUS) * sizeof(float);
    
    // создаем события для измерения времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // запускаем простое ядро
    std::cout << "\nзапуск простой свёртки...\n";
    cudaEventRecord(start);
    convolutionSimple<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, d_kernel, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "время простой свёртки: " << milliseconds << " мс\n";
    
    // запускаем оптимизированное ядро
    std::cout << "запуск оптимизированной свёртки...\n";
    cudaEventRecord(start);
    convolutionShared<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_input, d_output, d_kernel, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "время оптимизированной свёртки: " << milliseconds << " мс\n";
    
    // копируем результат обратно
    cudaMemcpy(h_output, d_output, imageBytes, cudaMemcpyDeviceToHost);
    
    // проверяем несколько пикселей результата
    std::cout << "\nпроверка некоторых значений выходного изображения:\n";
    std::cout << "центральный пиксель: " << h_output[height/2 * width + width/2] << "\n";
    std::cout << "верхний левый: " << h_output[0] << "\n";
    std::cout << "нижний правый: " << h_output[height * width - 1] << "\n";
    
    // освобождаем память
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
    
    delete[] h_input;
    delete[] h_output;
    delete[] h_kernel;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    std::cout << "\nпрограмма завершена\n";
    return 0;
}
