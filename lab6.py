import numpy as np
import random

# Кодирование сообщения
def encode_data(u, g):
    """Функция для кодирования сообщения с использованием порождающего многочлена."""
    return np.polymul(u, g) % 2

# Внесение ошибок
def inject_errors(word, error_count):
    """Внесение заданного количества случайных ошибок в сообщение."""
    n = len(word)
    error_positions = random.sample(range(n), error_count)
    print(f"Ошибки внесены в позиции: {error_positions}")
    for pos in error_positions:
        word[pos] ^= 1  # Инвертируем бит
    return word

# Генерация пакета ошибок
def inject_error_packet(word, t):
    """Внесение пакета ошибок длиной t в случайные позиции сообщения."""
    n = len(word)
    start_pos = random.randint(0, n - t)
    for i in range(t):
        word[(start_pos + i) % n] ^= 1  # Инвертируем биты пакета
    print(f"Пакет ошибок внесён в позиции от {start_pos} до {(start_pos + t - 1) % n}")
    return word

# Декодирование сообщения
def is_error_valid(error, t):
    """Проверка, является ли ошибка допустимой по длине (не длиннее t)."""
    error = np.trim_zeros(error, 'f')  # Убираем ведущие нули
    error = np.trim_zeros(error, 'b')  # Убираем конечные нули
    return len(error) <= t and len(error) != 0

def decode_data(w, g, t, is_packet):
    """Декодирование сообщения с исправлением одиночных или пакетов ошибок."""
    n = len(w)
    s = np.polydiv(w, g)[1] % 2  # Остаток от деления (синдром)

    for i in range(n):
        e_x = np.zeros(n, dtype=int)
        e_x[n - i - 1] = 1
        mult = np.polymul(s, e_x) % 2

        s_i = np.polydiv(mult, g)[1] % 2

        if is_packet:
            if is_error_valid(s_i, t):
                e_i = np.zeros(n, dtype=int)
                e_i[i - 1] = 1
                e_x = np.polymul(e_i, s_i) % 2
                corrected = np.polyadd(e_x, w) % 2
                result = np.array(np.polydiv(corrected, g)[0] % 2).astype(int)
                return result
        else:
            if sum(s_i) <= t:
                e_i = np.zeros(n, dtype=int)
                e_i[i - 1] = 1
                e_x = np.polymul(e_i, s_i) % 2
                corrected = np.polyadd(e_x, w) % 2
                result = np.array(np.polydiv(corrected, g)[0] % 2).astype(int)
                return result
    return None

# 6.1 Исследование кода (7, 4)
def test_code_7_4():
    """Исследование кодирования и исправления ошибок для кода (7, 4)."""
    print("-------------------------------\nИсследование кода (7, 4)\n")
    g = np.array([1, 1, 0, 1])  # Порождающий многочлен
    t = 1  # Допустимая кратность ошибки

    for error_count in range(1, 4):
        word = np.array([1, 0, 1, 0])  # Исходное сообщение
        print(f"Исходное сообщение: {word}")
        codeword = encode_data(word, g)  # Кодируем сообщение
        print(f"Закодированное сообщение: {codeword}")

        # Внесение ошибок
        codeword_with_errors = inject_errors(codeword.copy(), error_count)
        print(f"Синдром: {codeword_with_errors}")

        # Декодирование с исправлением ошибок
        decoded = decode_data(codeword_with_errors, g, t, is_packet=False)
        print(f"Декодированное сообщение: {decoded}")

        # Проверка совпадения исходного и декодированного сообщения
        if np.array_equal(word, decoded):
            print("Исходное сообщение и декодированное совпадают.\n")
        else:
            print("Исходное сообщение и декодированное не совпадают.\n")

# 6.2 Исследование кода (15, 9)
def test_code_15_9():
    """Исследование кодирования и исправления пакетов ошибок для кода (15, 9)."""
    print("-------------------------------\nИсследование кода (15, 9)\n")
    g = np.array([1, 0, 0, 1, 1, 1, 1])  # Порождающий многочлен
    t = 3  # Допустимая длина пакета ошибок

    for packet_length in range(1, 5):
        word = np.array([1, 1, 0, 0, 0, 1, 0, 0, 0])  # Исходное сообщение
        print(f"Исходное сообщение: {word}")
        codeword = encode_data(word, g)  # Кодируем сообщение
        print(f"Закодированное сообщение: {codeword}")

        # Внесение пакета ошибок
        codeword_with_packet_errors = inject_error_packet(codeword.copy(), packet_length)
        print(f"Синдром: {codeword_with_packet_errors}")

        # Декодирование с исправлением ошибок
        decoded = decode_data(codeword_with_packet_errors, g, t, is_packet=True)
        print(f"Декодированное сообщение: {decoded}")

        # Проверка совпадения исходного и декодированного сообщения
        if np.array_equal(word, decoded):
            print("Исходное и декодированное совпадают.\n")
        else:
            print("Исходное и декодированное не совпадают.\n")


# Основная часть программы
if __name__ == '__main__':
    test_code_7_4()  # Исследуем код (7, 4)
    test_code_15_9()  # Исследуем код (15, 9)
