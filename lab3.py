import numpy as np

# 3.1 Формирование порождающей и проверочной матриц кода Хэмминга
def hamming_matrices(r):
    n = 2**r - 1  # Длина кодового слова
    k = n - r  # Длина информационного слова

    # Формируем проверочную матрицу H
    H = []
    for i in range(1, n + 1):
        H.append(list(map(int, bin(i)[2:].zfill(r))))  # Двоичное представление чисел от 1 до n
    H = np.array(H).T

    # Формируем порождающую матрицу G
    I_k = np.eye(k, dtype=int)
    G = np.hstack((I_k, H[:, :k].T))  # Берём первые k столбцов из H, G = [I_k | H[:,:k].T]

    return G, H
# Формируем таблицу синдромов для однократных ошибок
def syndrome_table(H):
    n = H.shape[1]
    syndromes = {}
    for i in range(n):
        error_vector = np.zeros(n, dtype=int)
        error_vector[i] = 1
        syndrome = np.dot(error_vector, H.T) % 2
        syndromes[tuple(syndrome)] = error_vector
    return syndromes
r = 3
G, H = hamming_matrices(r)
syndromes = syndrome_table(H)

print(f"3.1 Порождающая матрица для r={r}:\n", G)
print(f"3.1 Проверочная матрица для r={r}:\n", H)
print(f"3.1 Таблица синдромов для r={r}:")
for syndrome, error in syndromes.items():
    print(f"Синдром {syndrome} -> Ошибка {error}")

# 3.2 Функции для кодирования и декодирования
def encode_hamming(information_word, G):
    return np.dot(information_word, G) % 2

def decode_hamming(code_word, H, syndromes):
    syndrome = np.dot(code_word, H.T) % 2
    if tuple(syndrome) in syndromes:
        error = syndromes[tuple(syndrome)]
        corrected_word = (code_word + error) % 2
        return corrected_word, True
    return code_word, False


# Функция для исследования одно-, двух- и трёхкратных ошибок
def hamming_error_investigation(r):
    G, H = hamming_matrices(r)
    syndromes = syndrome_table(H)

    information_word = np.random.randint(0, 2, size=G.shape[0])  # Случайное информационное слово
    code_word = encode_hamming(information_word, G)

    print(f"\n3.2 Исследование кода Хэмминга для r={r}")

    # Длина кодового слова n
    n = len(code_word)

    # Однократная ошибка (вносим только если n > 1)
    error = np.zeros(n, dtype=int)
    if n > 1:
        error[1] = 1  # Вносим однократную ошибку
        code_word_with_error = (code_word + error) % 2
        corrected_word, corrected = decode_hamming(code_word_with_error, H, syndromes)
        print(f"Однократная ошибка внесена: {code_word_with_error}")
        print(f"Исправлено: {corrected_word}, успешно ли исправлено: {'Да' if corrected else 'Нет'}")

    # Двукратная ошибка (вносим только если n > 3)
    if n > 3:
        error[3] = 1  # Вносим двукратную ошибку
        code_word_with_error = (code_word + error) % 2
        corrected_word, corrected = decode_hamming(code_word_with_error, H, syndromes)
        print(f"Двукратная ошибка внесена: {code_word_with_error}")
        print(f"Исправлено: {corrected_word}, успешно ли исправлено: {'Да' if corrected else 'Нет'}")

    # Трёхкратная ошибка (вносим только если n > 5)
    if n > 5:
        error[5] = 1  # Вносим трёхкратную ошибку
        code_word_with_error = (code_word + error) % 2
        corrected_word, corrected = decode_hamming(code_word_with_error, H, syndromes)
        print(f"Трёхкратная ошибка внесена: {code_word_with_error}")
        print(f"Исправлено: {corrected_word}, успешно ли исправлено: {'Да' if corrected else 'Нет'}")

# Исследование для r = 2, 3, 4
for r in [2, 3, 4]:
    hamming_error_investigation(r)

# 3.3 Формирование порождающей и проверочной матриц расширенного кода Хэмминга
def extended_hamming_matrices(r):
    G, H = hamming_matrices(r)
    n = 2**r  # Длина кодового слова для расширенного кода Хэмминга
    # Расширяем порождающую матрицу G: добавляем столбец для дополнительного проверочного бита
    G_ext = np.hstack((G, np.ones((G.shape[0], 1), dtype=int)))  # Добавляем единичный столбец

    # Расширяем проверочную матрицу H: добавляем столбец для дополнительного проверочного бита
    H_ext = np.hstack((H, np.ones((H.shape[0], 1), dtype=int)))  # Добавляем единичный столбец

    return G_ext, H_ext

# 3.4 Исследование расширенного кода Хэмминга
def extended_hamming_error_investigation(r):
    G_ext, H_ext = extended_hamming_matrices(r)
    syndromes_ext = syndrome_table(H_ext)

    information_word = np.random.randint(0, 2, size=G_ext.shape[0])  # Случайное информационное слово
    code_word = encode_hamming(information_word, G_ext)

    # Вывод таблицы синдромов для всех однократных ошибок
    print(f"3.3 Таблица синдромов для расширенного кода Хэмминга (r={r}):")
    for syndrome, error in syndromes_ext.items():
        print(f"Синдром {syndrome} -> Ошибка {error}")
    # Добавляем пробел для визуального отделения таблицы синдромов
    print("\n3.4 Исследование кратных ошибок:")

    # Длина кодового слова n
    n = len(code_word)

    # Однократная ошибка
    error = np.zeros(n, dtype=int)
    if n > 1:
        error[1] = 1
        code_word_with_error = (code_word + error) % 2
        corrected_word, corrected = decode_hamming(code_word_with_error, H_ext, syndromes_ext)
        print(f"Однократная ошибка: {code_word_with_error}, Исправлено: {'Да' if corrected else 'Нет'}")

    # Двукратная ошибка
    if n > 3:
        error[3] = 1
        code_word_with_error = (code_word + error) % 2
        corrected_word, corrected = decode_hamming(code_word_with_error, H_ext, syndromes_ext)
        print(f"Двукратная ошибка: {code_word_with_error}, Исправлено: {'Да' if corrected else 'Нет'}")

    # Трёхкратная ошибка
    if n > 5:
        error[5] = 1
        code_word_with_error = (code_word + error) % 2
        corrected_word, corrected = decode_hamming(code_word_with_error, H_ext, syndromes_ext)
        print(f"Трёхкратная ошибка: {code_word_with_error}, Исправлено: {'Да' if corrected else 'Нет'}")

    # Четырёхкратная ошибка
    if n > 6:
        error[6] = 1
        code_word_with_error = (code_word + error) % 2
        corrected_word, corrected = decode_hamming(code_word_with_error, H_ext, syndromes_ext)
        print(f"Четырёхкратная ошибка: {code_word_with_error}, Исправлено: {'Да' if corrected else 'Нет'}")

# Вызов 3.4 после 3.3 для r = 2, 3, 4
for r in [2, 3, 4]:
    extended_hamming_error_investigation(r)

