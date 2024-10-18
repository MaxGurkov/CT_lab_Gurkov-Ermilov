import numpy as np
# 2.1 Сформировать порождающую матрицу линейного кода (7, 4, 3)
def generate_matrix_7_4_3():
    I_k = np.eye(4, dtype=int)  # Единичная матрица 4x4
    X = np.array([[1, 1, 0],
                  [1, 0, 1],
                  [0, 1, 1],
                  [1, 1, 1]], dtype=int)  # Дополнение X

    G = np.hstack((I_k, X))  # Формирование порождающей матрицы G
    return G

G = generate_matrix_7_4_3()
print("2.1 Порождающая матрица (7, 4, 3):\n", G)

# 2.2 Сформировать проверочную матрицу на основе порождающей
def generate_check_matrix(G):
    X = G[:, 4:]  # Матрица X (последние 3 столбца G)
    X_T = X.T  # Транспонируем X
    I_n_k = np.eye(3, dtype=int)  # Единичная матрица 3x3

    H = np.hstack((X_T, I_n_k))  # Формируем проверочную матрицу H
    return H

H = generate_check_matrix(G)
print("2.2 Проверочная матрица H:\n", H)

# 2.3 Сформировать таблицу синдромов для однократных ошибок
def generate_syndrome_table(H):
    syndromes = {}
    for i in range(7):  # Всего 7 позиций для ошибок
        error_vector = np.zeros(7, dtype=int)
        error_vector[i] = 1  # Ошибка в i-й позиции
        syndrome = np.dot(error_vector, H.T) % 2  # Вычисление синдрома
        syndromes[tuple(syndrome)] = error_vector
    return syndromes

syndrome_table = generate_syndrome_table(H)
print("2.3 Таблица синдромов для однократных ошибок:")
for syndrome, error_vector in syndrome_table.items():
    print(f"Синдром {syndrome} -> Ошибка {error_vector}")

# 2.4 Сформировать кодовое слово, внести ошибку, вычислить синдром и исправить ошибку
def encode_word(information_word, G):
    return np.dot(information_word, G) % 2

def decode_word(code_word, H, syndrome_table):
    syndrome = np.dot(code_word, H.T) % 2  # Вычисление синдрома
    if tuple(syndrome) in syndrome_table:
        error = syndrome_table[tuple(syndrome)]
        corrected_word = (code_word + error) % 2
        return corrected_word, True
    else:
        return code_word, False

information_word = np.array([1, 0, 1, 1], dtype=int)
code_word = encode_word(information_word, G)

# Внесение ошибки в 2-ю позицию
error = np.array([0, 0, 1, 0, 0, 0, 0], dtype=int)
code_word_with_error = (code_word + error) % 2

print("2.4 Кодовое слово с ошибкой:", code_word_with_error)

# Декодирование и исправление ошибки
corrected_word, corrected = decode_word(code_word_with_error, H, syndrome_table)
if corrected:
    print("2.4 Ошибка исправлена, кодовое слово:", corrected_word)
else:
    print("2.4 Ошибка не найдена")

# 2.5 Сформировать кодовое слово длины n из слова длины k. Внести двукратную ошибку, вычислить синдром и убедиться, что слово отличается

# Внесение двукратной ошибки в 2-ю и 4-ю позиции
error_double = np.array([0, 1, 0, 1, 0, 0, 0], dtype=int)
code_word_with_double_error = (code_word + error_double) % 2

print("2.5 Кодовое слово с двукратной ошибкой:", code_word_with_double_error)

# Вычисление синдрома для кодового слова с двукратной ошибкой
corrected_word_double, corrected_double = decode_word(code_word_with_double_error, H, syndrome_table)
if corrected_double:
    print("2.5 Ошибка исправлена, кодовое слово:", corrected_word_double)
else:
    print("2.5 Ошибка не исправлена. Кодовое слово отличается от исходного.")

# Проверка на отличие от исходного
if np.array_equal(corrected_word_double, code_word):
    print("2.5 Кодовое слово исправлено и совпадает с отправленным.")
else:
    print("2.5 Кодовое слово исправлено, но отличается от отправленного.")
# Вторая часть лабораторной работы

# 2.6 Сформировать порождающую матрицу линейного кода (n, k, 5)
def generate_matrix_n_k_5():
    I_k = np.eye(11, dtype=int)  # Единичная матрица 11x11
    X = np.array([
        [1, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [0, 1, 1, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 1, 1, 1],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]], dtype=int)  # Дополнение X (удовлетворяет условиям d=5)

    G = np.hstack((I_k, X))  # Формирование порождающей матрицы G
    return G

G = generate_matrix_n_k_5()
print("2.6 Порождающая матрица (15, 11, 5):\n", G)

# 2.7 Сформировать проверочную матрицу на основе порождающей
def generate_check_matrix_n_k_5(G):
    X = G[:, 11:]  # Матрица X (последние 4 столбца G)
    X_T = X.T  # Транспонируем X
    I_n_k = np.eye(4, dtype=int)  # Единичная матрица 4x4

    H = np.hstack((X_T, I_n_k))  # Формируем проверочную матрицу H
    return H

H = generate_check_matrix_n_k_5(G)
print("2.7 Проверочная матрица H:\n", H)

# 2.8 Сформировать таблицу синдромов для однократных и двукратных ошибок
def generate_syndrome_table_double_errors(H):
    syndromes = {}
    n = H.shape[1]  # Длина кодового слова
    # Однократные ошибки
    for i in range(n):
        error_vector = np.zeros(n, dtype=int)
        error_vector[i] = 1
        syndrome = np.dot(error_vector, H.T) % 2
        syndromes[tuple(syndrome)] = error_vector

    # Двукратные ошибки
    for i in range(n):
        for j in range(i + 1, n):
            error_vector = np.zeros(n, dtype=int)
            error_vector[i] = 1
            error_vector[j] = 1
            syndrome = np.dot(error_vector, H.T) % 2
            syndromes[tuple(syndrome)] = error_vector
    return syndromes

syndrome_table = generate_syndrome_table_double_errors(H)
print("2.8 Таблица синдромов для однократных и двукратных ошибок:")
for syndrome, error_vector in syndrome_table.items():
    print(f"Синдром {syndrome} -> Ошибка {error_vector}")

# 2.9 Сформировать кодовое слово, внести однократную ошибку и исправить её
information_word = np.array([1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1], dtype=int)
code_word = encode_word(information_word, G)

# Внесение ошибки в 3-ю позицию
error = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int)
code_word_with_error = (code_word + error) % 2

print("2.9 Кодовое слово с однократной ошибкой:", code_word_with_error)

# Декодирование и исправление ошибки
corrected_word, corrected = decode_word(code_word_with_error, H, syndrome_table)
if corrected:
    print("2.9 Ошибка исправлена, кодовое слово:", corrected_word)
else:
    print("2.9 Ошибка не найдена")

# 2.10 Внесение двукратной ошибки и исправление
error_double = np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int)
code_word_with_double_error = (code_word + error_double) % 2

print("2.10 Кодовое слово с двукратной ошибкой:", code_word_with_double_error)

corrected_word_double, corrected_double = decode_word(code_word_with_double_error, H, syndrome_table)
if corrected_double:
    print("2.10 Ошибка исправлена, кодовое слово:", corrected_word_double)
else:
    print("2.10 Ошибка не найдена")

# 2.11 Внесение трёхкратной ошибки и проверка
error_triple = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int)
code_word_with_triple_error = (code_word + error_triple) % 2

print("2.11 Кодовое слово с трёхкратной ошибкой:", code_word_with_triple_error)

corrected_word_triple, corrected_triple = decode_word(code_word_with_triple_error, H, syndrome_table)
if corrected_triple:
    print("2.11 Ошибка исправлена, кодовое слово:", corrected_word_triple)
else:
    print("2.11 Ошибка не найдена. Кодовое слово отличается от отправленного.")
