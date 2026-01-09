// ============================================================================
// Практическая работа №2
// Параллельная реализация простых алгоритмов сортировки с OpenMP
// ============================================================================

#include <iostream>      // cout, cin
#include <vector>        // vector
#include <chrono>        // chrono для времени
#include <random>        // генерация случайных чисел
#include <algorithm>     // swap, is_sorted
#include <limits>        // numeric_limits

#ifdef _OPENMP           // если OpenMP включён
#include <omp.h>         // функции OpenMP
#endif

using namespace std;     // используем std по умолчанию

// ============================================================================
// Генерация массива случайных чисел
// ============================================================================

vector<int> generateArray(int n) {                       // создаёт массив из n элементов
    vector<int> a(n);                                    // выделяем n элементов
    random_device rd;                                    // источник случайности
    mt19937 gen(rd());                                   // генератор mt19937, seed от rd()
    uniform_int_distribution<int> dist(0, 10000);         // числа в диапазоне 0..10000

    for (int i = 0; i < n; ++i) {                        // идём по всем индексам
        a[i] = dist(gen);                                // кладём случайное число
    }                                                    // конец заполнения

    return a;                                            // возвращаем массив
}                                                        // конец generateArray

// ============================================================================
// 1) Bubble Sort — последовательная версия
// ============================================================================

void bubbleSortSequential(vector<int>& a) {              // пузырёк: последовательная сортировка
    int n = (int)a.size();                               // размер массива

    for (int i = 0; i < n - 1; ++i) {                    // количество проходов
        for (int j = 0; j < n - i - 1; ++j) {            // сравнение соседних
            if (a[j] > a[j + 1]) {                       // если пара стоит неправильно
                swap(a[j], a[j + 1]);                    // меняем местами
            }                                            // конец if
        }                                                // конец внутреннего цикла
    }                                                    // конец внешнего цикла
}                                                        // конец bubbleSortSequential

// ============================================================================
// 1) Bubble Sort — параллельная версия (odd-even)
// ВАЖНО: такой пузырёк работает фазами, чтобы не было конфликтов соседних swap
// ============================================================================

void bubbleSortParallel(vector<int>& a) {                // пузырёк: параллельная версия odd-even
    int n = (int)a.size();                               // размер массива

    for (int phase = 0; phase < n; ++phase) {            // делаем n фаз
        int start = (phase % 2 == 0) ? 0 : 1;             // чётная фаза: 0, нечётная: 1

#ifdef _OPENMP
        #pragma omp parallel for schedule(static)         // делим пары между потоками равномерно
#endif
        for (int j = start; j < n - 1; j += 2) {          // обрабатываем пары (j, j+1)
            if (a[j] > a[j + 1]) {                        // если пара не по порядку
                swap(a[j], a[j + 1]);                     // меняем
            }                                             // конец if
        }                                                 // конец omp for
    }                                                     // конец фаз
}                                                         // конец bubbleSortParallel

// ============================================================================
// 2) Selection Sort — последовательная версия
// ============================================================================

void selectionSortSequential(vector<int>& a) {            // выбором: последовательная сортировка
    int n = (int)a.size();                                // размер массива

    for (int i = 0; i < n - 1; ++i) {                     // позиция, куда ставим минимум
        int minIndex = i;                                 // считаем текущий минимум = i

        for (int j = i + 1; j < n; ++j) {                 // ищем минимум справа
            if (a[j] < a[minIndex]) {                     // если нашли меньше
                minIndex = j;                             // обновили индекс минимума
            }                                             // конец if
        }                                                 // конец поиска минимума

        swap(a[i], a[minIndex]);                          // ставим минимум на позицию i
    }                                                     // конец внешнего цикла
}                                                         // конец selectionSortSequential

// ============================================================================
// 2) Selection Sort — параллельная версия
// Внешний цикл i последовательный, а поиск минимума внутри — параллельный
// ============================================================================

void selectionSortParallel(vector<int>& a) {              // выбором: параллельный поиск минимума
    int n = (int)a.size();                                // размер массива

    for (int i = 0; i < n - 1; ++i) {                     // фиксируем позицию i
        int globalMinVal = a[i];                          // глобальный минимум по значению
        int globalMinIdx = i;                             // глобальный минимум по индексу

#ifdef _OPENMP
        #pragma omp parallel                               // создаём параллельную область один раз на i
        {
            int localMinVal = globalMinVal;                // локальный минимум потока
            int localMinIdx = globalMinIdx;                // локальный индекс минимума

            #pragma omp for nowait schedule(static)         // делим диапазон [i+1..n-1] между потоками
            for (int j = i + 1; j < n; ++j) {              // каждый поток смотрит свою часть
                if (a[j] < localMinVal) {                  // если нашли меньше
                    localMinVal = a[j];                    // запоминаем значение
                    localMinIdx = j;                       // запоминаем индекс
                }                                          // конец if
            }                                              // конец for

            #pragma omp critical                            // один поток за раз обновляет глобальный минимум
            {
                if (localMinVal < globalMinVal) {          // если локальный минимум лучше глобального
                    globalMinVal = localMinVal;            // обновляем глобальное значение
                    globalMinIdx = localMinIdx;            // обновляем глобальный индекс
                }                                          // конец if
            }                                              // конец critical
        }                                                  // конец parallel
#else
        for (int j = i + 1; j < n; ++j) {                  // последовательный поиск (если OpenMP нет)
            if (a[j] < globalMinVal) {                     // нашли меньше
                globalMinVal = a[j];                       // обновили минимум
                globalMinIdx = j;                          // обновили индекс
            }                                              // конец if
        }                                                  // конец for
#endif

        swap(a[i], a[globalMinIdx]);                       // ставим минимум на позицию i
    }                                                      // конец внешнего цикла
}                                                          // конец selectionSortParallel

// ============================================================================
// 3) Insertion Sort — последовательная версия
// ============================================================================

void insertionSortSequential(vector<int>& a) {             // вставками: последовательная сортировка
    int n = (int)a.size();                                 // размер массива

    for (int i = 1; i < n; ++i) {                          // начинаем со 2-го элемента
        int key = a[i];                                    // запоминаем вставляемый элемент
        int j = i - 1;                                     // идём влево от i

        while (j >= 0 && a[j] > key) {                     // пока слева элементы больше key
            a[j + 1] = a[j];                               // сдвигаем элемент вправо
            --j;                                           // двигаемся дальше влево
        }                                                  // конец while

        a[j + 1] = key;                                    // вставляем key на правильное место
    }                                                      // конец for
}                                                          // конец insertionSortSequential

// ============================================================================
// 3) Insertion Sort — "параллельная" версия
// Реально insertion плохо параллелится: шаг i зависит от предыдущих шагов.
// Вариант с critical не ускоряет алгоритм честно, поэтому делаем безопасно:
// просто вызываем последовательную версию.
// ============================================================================

void insertionSortParallel(vector<int>& a) {               // вставками: параллельная версия по заданию
    insertionSortSequential(a);                             // честно выполняем вставки последовательно
}                                                          // конец insertionSortParallel

// ============================================================================
// Измерение времени + проверка корректности
// Делает несколько прогонов и берёт минимальное время (снижает шум)
// ============================================================================

template <typename Func>
double measureSortBestOf3(const vector<int>& base, Func sortFunc, const string& name) {
    double bestMs = numeric_limits<double>::max();         // лучшее время (минимум)
    bool allOk = true;                                     // флаг корректности

    for (int run = 0; run < 3; ++run) {                    // три запуска
        vector<int> a = base;                               // каждый раз сортируем одинаковые данные

        auto start = chrono::high_resolution_clock::now();  // время начала
        sortFunc(a);                                        // сортировка
        auto end = chrono::high_resolution_clock::now();    // время конца

        double ms = chrono::duration<double, milli>(end - start).count(); // миллисекунды
        bestMs = min(bestMs, ms);                            // берём лучшее время

        bool ok = is_sorted(a.begin(), a.end());             // проверка, что отсортировано
        allOk = allOk && ok;                                 // если хоть раз сломалось — будет false
    }                                                        // конец трёх прогонов

    cout << name << ": " << bestMs << " ms";                 // печатаем лучшее время
    cout << " (" << (allOk ? "OK" : "FAIL") << ")";          // печатаем, правильно ли отсортировалось
    cout << "\n";                                            // перевод строки

    return bestMs;                                           // возвращаем лучшее время
}

// ============================================================================
// Запуск тестов на одном размере массива
// ============================================================================

void runTest(int n) {                                       // запускаем тесты для размера n
    cout << "\nArray size: " << n << "\n";                  // печатаем размер массива

    vector<int> base = generateArray(n);                    // создаём исходный массив

    measureSortBestOf3(base, bubbleSortSequential,  "Bubble Sequential");   // пузырёк sequential
    measureSortBestOf3(base, bubbleSortParallel,    "Bubble Parallel");     // пузырёк parallel

    measureSortBestOf3(base, selectionSortSequential,"Selection Sequential");// выбор sequential
    measureSortBestOf3(base, selectionSortParallel,  "Selection Parallel"); // выбор parallel

    measureSortBestOf3(base, insertionSortSequential,"Insertion Sequential");// вставки sequential
    measureSortBestOf3(base, insertionSortParallel,  "Insertion Parallel"); // вставки parallel
}

// ============================================================================
// main
// ============================================================================

int main() {                                                // точка входа
    cout << "PARALLEL SORTING WITH OpenMP\n";               // заголовок

#ifdef _OPENMP
    cout << "OpenMP enabled, threads: "                     // печатаем, что OpenMP включён
         << omp_get_max_threads() << "\n";                  // печатаем число доступных потоков
#else
    cout << "OpenMP not enabled\n";                         // если OpenMP нет
#endif

    runTest(1000);                                          // тест на 1000
    runTest(10000);                                         // тест на 10000
    runTest(100000);                                        // тест на 100000

    cout << "\nDone.\n";                                    // финальное сообщение
    return 0;                                               // успешный выход
}                                                           // конец main
