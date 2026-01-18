#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <omp.h>

using namespace std;

int main() {
    // задаем размер массива - 5 миллионов элементов
    const int n = 5000000;
    
    // выделяем динамически память под массив из 5 миллионов чисел
    int* arr = new int[n];
    
    // инициализируем генератор случайных чисел текущим временем
    srand(time(0));
    
    // заполняем массив случайными числами от 1 до 100
    for (int i = 0; i < n; i++) {
        // генерируем случайное число от 1 до 100 и записываем в массив
        arr[i] = rand() % 100 + 1;
    }
    
    // --- последовательное вычисление среднего ---
    
    // запоминаем время начала последовательного алгоритма
    auto seq_start = chrono::high_resolution_clock::now();
    
    // инициализируем сумму нулем
    long long seq_sum = 0;
    
    // последовательно проходим по массиву и суммируем элементы
    for (int i = 0; i < n; i++) {
        // прибавляем текущий элемент к общей сумме
        seq_sum += arr[i];
    }
    
    // вычисляем среднее значение делением суммы на количество элементов
    double seq_avg = (double)seq_sum / n;
    
    // фиксируем время окончания последовательного алгоритма
    auto seq_end = chrono::high_resolution_clock::now();
    
    // вычисляем длительность выполнения последовательной версии
    auto seq_duration = chrono::duration_cast<chrono::microseconds>(seq_end - seq_start);
    
    // --- параллельное вычисление среднего с openmp ---
    
    // запоминаем время начала параллельного алгоритма
    auto par_start = chrono::high_resolution_clock::now();
    
    // инициализируем сумму нулем
    long long par_sum = 0;
    
    // используем openmp для параллельного суммирования элементов массива
    // reduction(+:par_sum) - каждый поток будет считать свою частичную сумму
    // потом все частичные суммы автоматически сложатся в par_sum
    #pragma omp parallel for reduction(+:par_sum)
    for (int i = 0; i < n; i++) {
        // каждый поток обрабатывает свою часть массива
        // прибавляем элемент к локальной сумме потока
        par_sum += arr[i];
    }
    // после цикла openmp автоматически объединяет суммы всех потоков
    
    // вычисляем среднее значение делением суммы на количество элементов
    double par_avg = (double)par_sum / n;
    
    // фиксируем время окончания параллельного алгоритма
    auto par_end = chrono::high_resolution_clock::now();
    
    // вычисляем длительность выполнения параллельной версии
    auto par_duration = chrono::duration_cast<chrono::microseconds>(par_end - par_start);
    
    // вычисляем ускорение - во сколько раз параллельная версия быстрее
    double speedup = (double)seq_duration.count() / par_duration.count();
    
    // выводим результаты на экран
    cout << "=== Последовательное вычисление среднего ===" << endl;
    cout << "Сумма: " << seq_sum << endl;
    cout << "Среднее значение: " << seq_avg << endl;
    cout << "Время: " << seq_duration.count() << " микросекунд" << endl;
    cout << endl;
    
    cout << "=== Параллельное вычисление с OpenMP ===" << endl;
    cout << "Сумма: " << par_sum << endl;
    cout << "Среднее значение: " << par_avg << endl;
    cout << "Время: " << par_duration.count() << " микросекунд" << endl;
    cout << endl;
    
    cout << "=== Сравнение ===" << endl;
    cout << "Ускорение: " << speedup << "x" << endl;
    
    // проверяем что результаты совпадают
    cout << "Результаты совпадают: " << (seq_sum == par_sum ? "Да" : "Нет") << endl;
    
    // освобождаем выделенную память
    delete[] arr;
    
    return 0;
}