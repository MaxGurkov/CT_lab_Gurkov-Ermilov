import numpy as np
from itertools import combinations

# 4.1 Функция для формирования порождающей матрицы кода Рида-Маллера в каноническом виде
def reed_muller_generator(r, m):
    n = 2 ** m  # количество позиций в слове
    G = []

    # Стандартный порядок позиций вектора длиной n (двоичное представление)
    positions = [format(i, f'0{m}b') for i in range(n)]

    # Формирование базисных векторов для подмножеств до размера r включительно
    for k in range(r + 1):
        for subset in combinations(range(m), k):
            row = []
            # Построение строки для текущего подмножества
            for pos in positions:
                value = 1
                for idx in subset:
                    if pos[idx] == '0':
                        value = 0
                        break
                row.append(value)
            G.append(row)

    return np.array(G, dtype=int)

# Пример формирования RM(2, 4)
G_rm_2_4 = reed_muller_generator(2, 4)
print("4.1 Порождающая матрица для RM(2, 4):\n", G_rm_2_4)

# 4.2 Алгоритм мажоритарного декодирования для кода Рида-Маллера
def majority_decode(received_word, r, m):
    n = 2 ** m  # Длина кодового слова
    decoded_word = np.copy(received_word)

    # Основной цикл декодирования, от старшего уровня r до 0
    for i in range(r, -1, -1):
        for subset in combinations(range(m), i):
            count_1 = 0
            count_0 = 0
            # Подсчёт числа единиц и нулей для текущего подмножества позиций
            for j in range(n):
                match = all((received_word[j] == 0) == (j >> bit) % 2 == 0 for bit in subset)
                if match:
                    if received_word[j] == 1:
                        count_1 += 1
                    else:
                        count_0 += 1

            majority_vote = 1 if count_1 > count_0 else 0

            for j in range(n):
                match = all((j >> bit) % 2 == 0 for bit in subset)
                if match:
                    decoded_word[j] = majority_vote

    return decoded_word

# 4.3 Проверка алгоритма для RM(2, 4)
# Создаём случайное слово длиной 16 для проверки декодирования
test_word = np.random.randint(0, 2, 16)
print("4.2 Принятое слово:", test_word)
decoded_word = majority_decode(test_word, 2, 4)
print("4.3 Декодированное слово:", decoded_word)
