\# Assignment 2 - Гетерогенная параллелизация



Полное руководство по компиляции и запуску всех заданий на разных платформах.



\## Содержание



\- \[Требования](#требования)

\- \[Структура проекта](#структура-проекта)

\- \[Windows](#windows)

\- \[Linux](#linux)

\- \[macOS](#macos)

\- \[Проверка результатов](#проверка-результатов)

\- \[Решение проблем](#решение-проблем)



\## Требования



\### Общие требования



\- Компилятор C++ с поддержкой C++11 или выше

\- OpenMP (обычно идет с компилятором)

\- Для задачи 4: NVIDIA GPU и CUDA Toolkit



\### Версии



\- g++ 7.0+ или clang 3.9+ или MSVC 2017+

\- OpenMP 4.0+

\- CUDA Toolkit 10.0+ (только для задачи 4)



\## Структура проекта



```

assignment2/

├── task2\_minmax.cpp           # задача 2: поиск min/max с openmp

├── task3\_selection\_sort.cpp   # задача 3: сортировка выбором с openmp

├── task4\_merge\_sort\_gpu.cu    # задача 4: сортировка на gpu с cuda

├── theory.md                  # теоретическая часть

├── control\_questions.md       # ответы на контрольные вопросы

└── README.md                  # этот файл

```



---



\## Windows



\### Установка необходимых инструментов



\#### Вариант 1: Visual Studio (рекомендуется)



1\. Скачайте Visual Studio Community (бесплатно): https://visualstudio.microsoft.com/

2\. При установке выберите:

&nbsp;  - Desktop development with C++

&nbsp;  - CUDA development (если есть NVIDIA GPU)

3\. OpenMP включен по умолчанию



\#### Вариант 2: MinGW-w64



1\. Скачайте MinGW-w64: https://www.mingw-w64.org/

2\. Установите и добавьте в PATH

3\. OpenMP включен по умолчанию



\#### CUDA Toolkit (для задачи 4)



1\. Скачайте с https://developer.nvidia.com/cuda-downloads

2\. Установите, следуя инструкциям

3\. Проверьте установку: `nvcc --version`



\### Компиляция и запуск



\#### Задача 2 (MinGW)



```bash

\# компиляция

g++ -fopenmp task2\_minmax.cpp -o task2\_minmax.exe



\# запуск

task2\_minmax.exe

```



\#### Задача 2 (Visual Studio)



```bash

\# компиляция из командной строки developer command prompt

cl /EHsc /openmp task2\_minmax.cpp



\# запуск

task2\_minmax.exe

```



\#### Задача 3 (MinGW)



```bash

\# компиляция

g++ -fopenmp task3\_selection\_sort.cpp -o task3\_selection\_sort.exe



\# запуск

task3\_selection\_sort.exe

```



\#### Задача 3 (Visual Studio)



```bash

\# компиляция

cl /EHsc /openmp task3\_selection\_sort.cpp



\# запуск

task3\_selection\_sort.exe

```



\#### Задача 4 (CUDA)



```bash

\# компиляция

nvcc task4\_merge\_sort\_gpu.cu -o task4\_merge\_sort\_gpu.exe



\# запуск

task4\_merge\_sort\_gpu.exe

```



\### Использование Visual Studio IDE



1\. Откройте Visual Studio

2\. File -> New -> Project

3\. Выберите "Empty Project"

4\. Добавьте файл: Project -> Add Existing Item

5\. Для OpenMP: Project Properties -> C/C++ -> Language -> OpenMP Support -> Yes

6\. Для CUDA: добавьте .cu файл, VS автоматически определит CUDA

7\. Build -> Build Solution (Ctrl+Shift+B)

8\. Debug -> Start Without Debugging (Ctrl+F5)



---



\## Linux



\### Установка необходимых инструментов



\#### Ubuntu/Debian



```bash

\# обновление репозиториев

sudo apt update



\# установка компилятора с openmp

sudo apt install build-essential



\# проверка что openmp работает

echo '#include <omp.h>

int main() { return omp\_get\_max\_threads(); }' > test.cpp

g++ -fopenmp test.cpp \&\& ./a.out \&\& echo "openmp работает"

rm test.cpp a.out

```



\#### Установка CUDA (для задачи 4)



```bash

\# добавление репозитория nvidia

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86\_64/cuda-keyring\_1.0-1\_all.deb

sudo dpkg -i cuda-keyring\_1.0-1\_all.deb

sudo apt update



\# установка cuda

sudo apt install cuda



\# добавление в path

echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc

echo 'export LD\_LIBRARY\_PATH=/usr/local/cuda/lib64:$LD\_LIBRARY\_PATH' >> ~/.bashrc

source ~/.bashrc



\# проверка

nvcc --version

```



\### Компиляция и запуск



\#### Задача 2



```bash

\# компиляция

g++ -fopenmp -O2 task2\_minmax.cpp -o task2\_minmax



\# запуск

./task2\_minmax

```



\#### Задача 3



```bash

\# компиляция

g++ -fopenmp -O2 task3\_selection\_sort.cpp -o task3\_selection\_sort



\# запуск

./task3\_selection\_sort

```



\#### Задача 4



```bash

\# компиляция

nvcc -O2 task4\_merge\_sort\_gpu.cu -o task4\_merge\_sort\_gpu



\# запуск

./task4\_merge\_sort\_gpu

```



\### Makefile (опционально)



Создайте файл `Makefile`:



```makefile

CXX = g++

CXXFLAGS = -fopenmp -O2 -std=c++11

NVCC = nvcc

NVCCFLAGS = -O2



all: task2 task3 task4



task2: task2\_minmax.cpp

&nbsp;	$(CXX) $(CXXFLAGS) $< -o task2\_minmax



task3: task3\_selection\_sort.cpp

&nbsp;	$(CXX) $(CXXFLAGS) $< -o task3\_selection\_sort



task4: task4\_merge\_sort\_gpu.cu

&nbsp;	$(NVCC) $(NVCCFLAGS) $< -o task4\_merge\_sort\_gpu



clean:

&nbsp;	rm -f task2\_minmax task3\_selection\_sort task4\_merge\_sort\_gpu



run\_all: all

&nbsp;	@echo "=== Задача 2 ==="

&nbsp;	./task2\_minmax

&nbsp;	@echo ""

&nbsp;	@echo "=== Задача 3 ==="

&nbsp;	./task3\_selection\_sort

&nbsp;	@echo ""

&nbsp;	@echo "=== Задача 4 ==="

&nbsp;	./task4\_merge\_sort\_gpu

```



Использование:



```bash

\# компиляция всех задач

make



\# запуск всех задач

make run\_all



\# очистка

make clean

```



---



\## macOS



\### Установка необходимых инструментов



\#### Установка Xcode и компилятора



```bash

\# установка xcode command line tools

xcode-select --install



\# установка homebrew (если нет)

/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"



\# установка gcc с openmp (clang от apple не поддерживает openmp полноценно)

brew install gcc



\# проверка версии

gcc-13 --version  # версия может отличаться

```



\#### Примечание по CUDA на macOS



Apple прекратила поддержку NVIDIA GPU начиная с macOS 10.14. Если у вас старый Mac с NVIDIA GPU и старая версия macOS, можно установить CUDA. Для современных Mac с Apple Silicon задача 4 невозможна без виртуальной машины.



\### Компиляция и запуск



\#### Задача 2



```bash

\# компиляция (используем gcc-13 вместо g++)

gcc-13 -fopenmp -O2 task2\_minmax.cpp -o task2\_minmax -lstdc++



\# запуск

./task2\_minmax

```



\#### Задача 3



```bash

\# компиляция

gcc-13 -fopenmp -O2 task3\_selection\_sort.cpp -o task3\_selection\_sort -lstdc++



\# запуск

./task3\_selection\_sort

```



\#### Задача 4



На современных Mac с Apple Silicon (M1/M2/M3) CUDA не работает. Варианты:



1\. Использовать Linux или Windows на другой машине

2\. Использовать удаленный сервер с GPU

3\. Использовать облачный сервис (Google Colab, AWS, etc)



\### Альтернатива: Docker



Можно использовать Docker для запуска Linux окружения:



```bash

\# установка docker desktop для mac

\# скачайте с https://www.docker.com/products/docker-desktop



\# запуск ubuntu контейнера

docker run -it --rm ubuntu:22.04 bash



\# внутри контейнера установите инструменты и следуйте инструкциям для linux

```



---



\## Проверка результатов



\### Задача 2



Ожидаемый вывод:



```

генерация массива из 10000 элементов...

массив создан



последовательная версия:

минимум: \[число]

максимум: \[число]

время: \[время] микросекунд



параллельная версия (openmp):

минимум: \[число]

максимум: \[число]

время: \[время] микросекунд



результаты совпадают - все верно!

ускорение: \[число]x



выводы:

\[автоматические выводы на основе результатов]

```



\### Задача 3



Ожидаемый вывод:



```

используется потоков: 4



===== тестирование для массива из 1000 элементов =====

последовательная сортировка:

время: \[время] мс

результат: верно



параллельная сортировка (openmp):

время: \[время] мс

результат: верно



результаты обеих версий совпадают

ускорение: \[число]x



\[аналогично для 10000 элементов]



===== выводы =====

\[автоматические выводы]

```



\### Задача 4



Ожидаемый вывод:



```

используется gpu: \[название gpu]

вычислительных блоков: \[число]



===== тестирование для массива из 10000 элементов =====

время сортировки на gpu: \[время] мс

результат: массив отсортирован верно

первые 5 элементов: \[числа]

последние 5 элементов: \[числа]



\[аналогично для 100000 элементов]



===== выводы =====

\[автоматические выводы]

```



---



\## Решение проблем



\### OpenMP не работает



\*\*Проблема:\*\* ошибка при компиляции с флагом `-fopenmp`



\*\*Решение для Windows:\*\*

\- Используйте Visual Studio вместо MinGW

\- Или установите более новую версию MinGW-w64



\*\*Решение для macOS:\*\*

\- Используйте gcc вместо clang: `gcc-13 -fopenmp ...`

\- Убедитесь что установлен gcc из homebrew



\*\*Решение для Linux:\*\*

\- Установите пакет `libomp-dev`: `sudo apt install libomp-dev`



\### CUDA ошибки



\*\*Проблема:\*\* `nvcc: command not found`



\*\*Решение:\*\*

\- Убедитесь что CUDA Toolkit установлен

\- Добавьте CUDA в PATH (см. инструкции выше)

\- Перезапустите терминал



\*\*Проблема:\*\* `cuda error: no CUDA-capable device is detected`



\*\*Решение:\*\*

\- Убедитесь что у вас NVIDIA GPU

\- Проверьте что драйвера установлены: `nvidia-smi`

\- На ноутбуках может быть два GPU, убедитесь что программа использует NVIDIA



\*\*Проблема:\*\* Программа компилируется но работает медленно



\*\*Решение:\*\*

\- Проверьте что программа использует GPU: `nvidia-smi` во время работы

\- Увеличьте размер массива для теста

\- Проверьте что не запущены другие программы использующие GPU



\### Программа падает с ошибкой памяти



\*\*Проблема:\*\* `Segmentation fault` или `Out of memory`



\*\*Решение:\*\*

\- Уменьшите размер массива

\- Проверьте что достаточно RAM (для CPU) или VRAM (для GPU)

\- Закройте другие программы



\### Медленная работа параллельных версий



\*\*Проблема:\*\* Параллельная версия медленнее последовательной



\*\*Причины:\*\*

\- Массив слишком маленький, накладные расходы больше выигрыша

\- Мало ядер в процессоре

\- Компилятор не оптимизировал код (добавьте флаг `-O2` или `-O3`)



\*\*Решение:\*\*

\- Увеличьте размер массива

\- Добавьте флаги оптимизации при компиляции

\- Это нормально для маленьких задач, так и должно быть



\### Разные результаты на разных запусках



\*\*Проблема:\*\* Время выполнения сильно различается



\*\*Причины:\*\*

\- Другие программы загружают процессор

\- Первый запуск может быть медленнее (холодный кеш)



\*\*Решение:\*\*

\- Закройте другие программы

\- Запустите программу несколько раз

\- Смотрите на среднее значение



---



\## Дополнительные опции компиляции



\### Уровни оптимизации



```bash

\# без оптимизации (для отладки)

g++ -fopenmp -O0 program.cpp



\# базовая оптимизация

g++ -fopenmp -O1 program.cpp



\# рекомендуемая оптимизация

g++ -fopenmp -O2 program.cpp



\# максимальная оптимизация (может увеличить время компиляции)

g++ -fopenmp -O3 program.cpp

```



\### Количество потоков OpenMP



Можно установить через переменную окружения:



```bash

\# windows

set OMP\_NUM\_THREADS=8

task2\_minmax.exe



\# linux/macos

export OMP\_NUM\_THREADS=8

./task2\_minmax

```



Или в самой программе (уже есть в коде):



```cpp

omp\_set\_num\_threads(4);  // установить 4 потока

```



\### Вывод отладочной информации CUDA



```bash

\# компиляция с отладочной информацией

nvcc -g -G task4\_merge\_sort\_gpu.cu -o task4\_merge\_sort\_gpu



\# профилирование с nvprof (старые версии cuda)

nvprof ./task4\_merge\_sort\_gpu



\# профилирование с nsys (новые версии cuda)

nsys profile ./task4\_merge\_sort\_gpu

```



---



\## Полезные ссылки



\- \[OpenMP документация](https://www.openmp.org/specifications/)

\- \[CUDA документация](https://docs.nvidia.com/cuda/)

\- \[GCC OpenMP](https://gcc.gnu.org/onlinedocs/libgomp/)

\- \[Visual Studio OpenMP](https://learn.microsoft.com/en-us/cpp/parallel/openmp/openmp-in-visual-cpp)



---



\## Контакты и помощь



Если возникли проблемы:



1\. Проверьте раздел "Решение проблем"

2\. Убедитесь что все требования установлены

3\. Проверьте версии компилятора и библиотек

4\. Попробуйте запустить на другой машине



