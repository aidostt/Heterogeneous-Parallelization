#include <mpi.h>
#include <iostream>
#include <vector>
#include <chrono>

// функция обработки локальной части массива (умножение на 2 и сумма)
double processLocalArray(float* local_data, int local_size) {
    double local_sum = 0.0;
    // проходим по локальной части и обрабатываем
    for (int i = 0; i < local_size; i++) {
        // умножаем на 2 и добавляем к сумме
        local_data[i] *= 2.0f;
        local_sum += local_data[i];
    }
    return local_sum;
}

int main(int argc, char** argv) {
    // инициализируем mpi
    MPI_Init(&argc, &argv);
    
    // получаем количество процессов
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // получаем ранг (номер) текущего процесса
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    // размер всего массива
    const int TOTAL_SIZE = 10000000;
    
    // вычисляем размер данных для каждого процесса
    int local_size = TOTAL_SIZE / world_size;
    // остаток достанется последнему процессу
    if (world_rank == world_size - 1) {
        local_size += TOTAL_SIZE % world_size;
    }
    
    // массив для хранения всех данных (только у процесса 0)
    std::vector<float> global_data;
    
    // процесс 0 инициализирует массив
    if (world_rank == 0) {
        global_data.resize(TOTAL_SIZE);
        // заполняем массив значениями
        for (int i = 0; i < TOTAL_SIZE; i++) {
            global_data[i] = static_cast<float>(i + 1);
        }
        std::cout << "Процессов: " << world_size << "\n";
        std::cout << "Размер массива: " << TOTAL_SIZE << " элементов\n";
    }
    
    // выделяем память для локальной части каждого процесса
    std::vector<float> local_data(local_size);
    
    // массив для указания размеров частей для каждого процесса
    std::vector<int> sendcounts;
    // массив для указания смещений для каждого процесса
    std::vector<int> displs;
    
    // процесс 0 готовит информацию о распределении
    if (world_rank == 0) {
        sendcounts.resize(world_size);
        displs.resize(world_size);
        
        int offset = 0;
        // вычисляем размеры и смещения для каждого процесса
        for (int i = 0; i < world_size; i++) {
            sendcounts[i] = TOTAL_SIZE / world_size;
            // последний процесс получает остаток
            if (i == world_size - 1) {
                sendcounts[i] += TOTAL_SIZE % world_size;
            }
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }
    
    // начинаем замер времени
    double start_time = MPI_Wtime();
    
    // распределяем данные между процессами
    // scatterv используется потому что размеры могут быть разными
    MPI_Scatterv(
        global_data.data(),      // данные для отправки (только у процесса 0)
        sendcounts.data(),       // массив размеров для каждого процесса
        displs.data(),           // массив смещений
        MPI_FLOAT,               // тип данных отправки
        local_data.data(),       // буфер для приёма локальных данных
        local_size,              // размер локальных данных
        MPI_FLOAT,               // тип данных приёма
        0,                       // корневой процесс (отправитель)
        MPI_COMM_WORLD           // коммуникатор
    );
    
    // каждый процесс обрабатывает свою локальную часть
    double local_sum = processLocalArray(local_data.data(), local_size);
    
    // собираем локальные суммы в глобальную сумму
    double global_sum = 0.0;
    MPI_Reduce(
        &local_sum,              // данные для отправки (локальная сумма)
        &global_sum,             // буфер для приёма результата
        1,                       // количество элементов
        MPI_DOUBLE,              // тип данных
        MPI_SUM,                 // операция (суммирование)
        0,                       // корневой процесс (получатель)
        MPI_COMM_WORLD           // коммуникатор
    );
    
    // собираем обработанные данные обратно в global_data
    MPI_Gatherv(
        local_data.data(),       // данные для отправки (локальная часть)
        local_size,              // размер локальных данных
        MPI_FLOAT,               // тип данных отправки
        global_data.data(),      // буфер для приёма (только у процесса 0)
        sendcounts.data(),       // массив размеров от каждого процесса
        displs.data(),           // массив смещений
        MPI_FLOAT,               // тип данных приёма
        0,                       // корневой процесс (получатель)
        MPI_COMM_WORLD           // коммуникатор
    );
    
    // заканчиваем замер времени
    double end_time = MPI_Wtime();
    
    // процесс 0 выводит результаты
    if (world_rank == 0) {
        std::cout << "Глобальная сумма: " << global_sum << "\n";
        std::cout << "Время выполнения: " << (end_time - start_time) * 1000.0 << " мс\n";
        
        // проверяем корректность (первые 5 элементов)
        std::cout << "Первые 5 обработанных элементов: ";
        for (int i = 0; i < 5; i++) {
            std::cout << global_data[i] << " ";
        }
        std::cout << "\n";
    }
    
    // завершаем mpi
    MPI_Finalize();
    
    return 0;
}

/* 
 * Компиляция: mpic++ -o task4 task4_mpi.cpp
 * Запуск с 2 процессами: mpirun -np 2 ./task4
 * Запуск с 4 процессами: mpirun -np 4 ./task4
 * Запуск с 8 процессами: mpirun -np 8 ./task4
 */
