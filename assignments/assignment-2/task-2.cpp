// задача 2: поиск минимума и максимума с openmp

#include <iostream>      // для вывода в консоль
#include <vector>        // для работы с динамическими массивами
#include <random>        // для генерации случайных чисел
#include <chrono>        // для замера времени
#include <omp.h>         // для openmp
#include <limits>        // для получения максимальных значений типов

using namespace std;

// функция для генерации массива случайных чисел
vector<int> generateArray(int size) {
    vector<int> arr(size);                          // создаем вектор нужного размера
    random_device rd;                               // инициализируем генератор случайных чисел
    mt19937 gen(rd());                              // используем алгоритм mersenne twister
    uniform_int_distribution<> dist(1, 10000);      // диапазон случайных чисел от 1 до 10000
    
    // заполняем массив случайными числами
    for (int i = 0; i < size; i++) {
        arr[i] = dist(gen);                         // генерируем случайное число и кладем в массив
    }
    
    return arr;                                     // возвращаем заполненный массив
}

// последовательный поиск минимума и максимума
void findMinMaxSequential(const vector<int>& arr, int& minVal, int& maxVal) {
    minVal = arr[0];                                // начальное значение минимума - первый элемент
    maxVal = arr[0];                                // начальное значение максимума - первый элемент
    
    // проходим по всем элементам массива
    for (size_t i = 1; i < arr.size(); i++) {
        if (arr[i] < minVal) {                      // если нашли число меньше текущего минимума
            minVal = arr[i];                        // обновляем минимум
        }
        if (arr[i] > maxVal) {                      // если нашли число больше текущего максимума
            maxVal = arr[i];                        // обновляем максимум
        }
    }
}

// параллельный поиск минимума и максимума с openmp
void findMinMaxParallel(const vector<int>& arr, int& minVal, int& maxVal) {
    minVal = numeric_limits<int>::max();            // начальное значение минимума - максимально возможное число
    maxVal = numeric_limits<int>::min();            // начальное значение максимума - минимально возможное число
    
    // параллельный цикл с редукцией
    // reduction(min:minVal) означает что каждый поток будет искать свой минимум
    // а потом все минимумы объединятся в один общий минимум
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < arr.size(); i++) {
        if (arr[i] < minVal) {                      // если нашли число меньше локального минимума
            minVal = arr[i];                        // обновляем локальный минимум
        }
        if (arr[i] > maxVal) {                      // если нашли число больше локального максимума
            maxVal = arr[i];                        // обновляем локальный максимум
        }
    }
    // после цикла openmp автоматически объединит результаты всех потоков
}

int main() {
    const int SIZE = 10000;                         // размер массива
    
    cout << "генерация массива из " << SIZE << " элементов..." << endl;
    vector<int> arr = generateArray(SIZE);          // создаем массив случайных чисел
    cout << "массив создан" << endl << endl;
    
    // последовательная версия
    int minSeq, maxSeq;                             // переменные для хранения результатов
    
    auto startSeq = chrono::high_resolution_clock::now();   // запоминаем время начала
    findMinMaxSequential(arr, minSeq, maxSeq);              // ищем минимум и максимум
    auto endSeq = chrono::high_resolution_clock::now();     // запоминаем время окончания
    
    // вычисляем время выполнения в микросекундах
    auto durationSeq = chrono::duration_cast<chrono::microseconds>(endSeq - startSeq);
    
    cout << "последовательная версия:" << endl;
    cout << "минимум: " << minSeq << endl;
    cout << "максимум: " << maxSeq << endl;
    cout << "время: " << durationSeq.count() << " микросекунд" << endl << endl;
    
    // параллельная версия
    int minPar, maxPar;                             // переменные для хранения результатов
    
    auto startPar = chrono::high_resolution_clock::now();   // запоминаем время начала
    findMinMaxParallel(arr, minPar, maxPar);                // ищем минимум и максимум параллельно
    auto endPar = chrono::high_resolution_clock::now();     // запоминаем время окончания
    
    // вычисляем время выполнения в микросекундах
    auto durationPar = chrono::duration_cast<chrono::microseconds>(endPar - startPar);
    
    cout << "параллельная версия (openmp):" << endl;
    cout << "минимум: " << minPar << endl;
    cout << "максимум: " << maxPar << endl;
    cout << "время: " << durationPar.count() << " микросекунд" << endl << endl;
    
    // проверяем что результаты совпадают
    if (minSeq == minPar && maxSeq == maxPar) {
        cout << "результаты совпадают - все верно!" << endl;
    } else {
        cout << "ошибка: результаты не совпадают!" << endl;
    }
    
    // выводим ускорение
    double speedup = (double)durationSeq.count() / durationPar.count();
    cout << "ускорение: " << speedup << "x" << endl << endl;
    
    // выводы
    cout << "выводы:" << endl;
    if (speedup > 1.5) {
        cout << "параллельная версия значительно быстрее последовательной" << endl;
        cout << "openmp эффективно распределил работу между потоками" << endl;
    } else if (speedup > 1.0) {
        cout << "параллельная версия немного быстрее" << endl;
        cout << "для такого размера массива накладные расходы на создание потоков заметны" << endl;
    } else {
        cout << "последовательная версия быстрее" << endl;
        cout << "массив слишком маленький для эффективной параллелизации" << endl;
    }
    
    return 0;
}