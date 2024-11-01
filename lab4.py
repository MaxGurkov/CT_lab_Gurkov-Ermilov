import numpy as np

# Матрица B для расширенного кода Голея
B = np.array([
    [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
    [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
    [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
    [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1],
    [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
    [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
    [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
], dtype=int)

# Формирование порождающей и проверочной матриц
def golay_matrices():
    I = np.eye(12, dtype=int)
    G = np.hstack((I, B))  # Порождающая матрица G = [I | B]
    H = np.vstack((I, B))  # Проверочная матрица H = [B; I]
    return G, H

G, H = golay_matrices()
print("4.1 Порождающая матрица G:\n", G)
print("4.1 Проверочная матрица H:\n", H)

import numpy as np

# Матрица B для расширенного кода Голея
B = np.array([
    [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
    [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
    [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
    [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1],
    [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
    [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
    [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
], dtype=int)

# Формирование порождающей и проверочной матриц для кода Голея
def golay_matrices():
    I = np.eye(12, dtype=int)
    G = np.hstack((I, B))  # Порождающая матрица G = [I, B]
    H = np.vstack((I, B))  # Проверочная матрица H = [I; B]
    return G, H

G, H = golay_matrices()

# Функция для кодирования информационного слова
def encode_golay(information_word, G):
    return np.dot(information_word, G) % 2

# Функция для вычисления синдрома
def calculate_syndrome(code_word, H):
    return (code_word @ H) % 2

# Функция для исправления ошибок на основе синдрома
def correct_errors(code_word, H):
    syndrome = calculate_syndrome(code_word, H)
    weight = np.sum(syndrome)

    if weight <= 3:  # Однократная ошибка или исправление на основе синдрома
        error_vector = np.zeros(len(code_word), dtype=int)
        error_vector[:len(syndrome)] = syndrome
        corrected_word = (code_word + error_vector) % 2
        return corrected_word, True
    elif weight > 3:  # Двухкратная и более сложные ошибки
        print(f"\Сложная ошибка с весом синдрома {weight}, невозможно исправить на основе синдрома.")
        return code_word, False

# Исследование одно-, двух-, трёх- и четырёхкратных ошибок
def golay_error_investigation(G, H):
    information_word = np.random.randint(0, 2, size=12)
    code_word = encode_golay(information_word, G)
    print(f"\n4.2 Исходное кодовое слово: {code_word}")

    for num_errors in range(1, 5):  # Одно-, двух-, трёх- и четырёхкратные ошибки
        error_positions = np.random.choice(len(code_word), num_errors, replace=False)
        corrupted_word = code_word.copy()
        for pos in error_positions:
            corrupted_word[pos] ^= 1  # Внесение ошибки

        print(f"\n{num_errors}-кратная ошибка на позициях {error_positions}: {corrupted_word}")
        corrected_word, corrected = correct_errors(corrupted_word, H)
        if corrected:
            print(f"Исправленное слово: {corrected_word}")
            print("Исправление успешно." if np.array_equal(corrected_word, code_word) else "Ошибка исправлена неверно.")
        else:
            print("Слово не поддаётся исправлению.")

# Выполнение исследования для кода Голея
golay_error_investigation(G, H)


# 4.3 Написать функцию формирования порождающей матрицы RM(r, m)
def reed_muller_generator(r, m):
    if r == 0:
        return np.ones((1, 2 ** m), dtype=int)
    elif r == m:
        return np.eye(2 ** m, dtype=int)
    else:
        G_rm1 = reed_muller_generator(r, m - 1)
        G_r1m1 = reed_muller_generator(r - 1, m - 1)
        upper = np.hstack((G_rm1, G_rm1))
        lower = np.hstack((np.zeros_like(G_r1m1), G_r1m1))
        return np.vstack((upper, lower))

# 4.3 Создание проверочной матрицы для RM(r, m)
def create_check_matrix_rm(G):
    n = G.shape[1]
    H_rows = []
    for i in range(n):
        row = np.zeros(n, dtype=int)
        row[i] = 1
        if (np.dot(row, G.T) % 2).all() == 0:  # Проверяем, ортогонально ли
            H_rows.append(row)
    return np.array(H_rows)

# 4.4. Провести исследование кода Рида-Маллера RM(1,3) для одно- и двукратных ошибок
# 4.5. Провести исследование кода Рида-Маллера RM(1,4) для одно-, двух-, трёх- и четырёхкратных ошибок
def introduce_errors_and_decode_rm(code_word, H, error_positions):
    corrupted_word = code_word.copy()
    for pos in error_positions:
        corrupted_word[pos] ^= 1  # Вносим ошибку
    syndrome = calculate_syndrome_rm(corrupted_word, H)
    print(f"Ошибки в позициях {error_positions}: Синдром = {syndrome}")

# Функция для вычисления синдрома
def calculate_syndrome_rm(code_word, H):
    return np.dot(code_word, H.T) % 2

# Исследование ошибок для RM(1, 3) и RM(1, 4)
def reed_muller_error_investigation(G, m):
    H = create_check_matrix_rm(G)  # Проверочная матрица для кода Рида-Маллера
    print(f"\nИсследование для RM(1, {m})")

    # Исходное кодовое слово (все нули)
    code_word = np.zeros(G.shape[1], dtype=int)

    # Однократная, двухкратная, трёхкратная и четырёхкратная ошибки
    for num_errors in range(1, 5):
        error_positions = np.random.choice(len(code_word), num_errors, replace=False)
        introduce_errors_and_decode_rm(code_word, H, error_positions)

# Основной код для исследования RM(1, 3) и RM(1, 4)
m = 3
print(f"\nПорождающая матрица Рида-Маллера (1,3): \n{reed_muller_generator(1, m)}\n")
reed_muller_error_investigation(reed_muller_generator(1, m), m)

m = 4
print(f"\nПорождающая матрица Рида-Маллера (1,4): \n{reed_muller_generator(1, m)}\n")
reed_muller_error_investigation(reed_muller_generator(1, m), m)
