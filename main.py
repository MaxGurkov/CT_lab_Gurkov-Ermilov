import numpy as np

# Определяем матрицу S как константу
S = np.array([[1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
              [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
              [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
              [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1],
              [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
              [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]])


def ref(matrix):
    """Приведение матрицы к ступенчатому виду"""
    edited_matr = matrix.copy()
    row, column = edited_matr.shape
    lead_elem_idx = 0
    for j in range(column):
        non_zero_row_idxs = np.nonzero(edited_matr[:, j] == 1)[0]
        if len(non_zero_row_idxs) > 0 and lead_elem_idx in non_zero_row_idxs:
            for i in range(lead_elem_idx + 1, row):
                if edited_matr[i][j] == 1:
                    edited_matr[i] = (edited_matr[i] + edited_matr[lead_elem_idx]) % 2
            lead_elem_idx += 1
    for i in range(row - 1, -1, -1):
        if np.all(edited_matr[i] == 0):
            edited_matr = np.delete(edited_matr, i, 0)
    return edited_matr


def rref(matrix):
    """Приведение матрицы к приведённому ступенчатому виду"""
    ref_matrix = ref(matrix)
    row = ref_matrix.shape[0]
    for i in range(row - 1, 0, -1):
        j = np.nonzero(ref_matrix[i] == 1)[0][0]
        for k in range(i - 1, -1, -1):
            if ref_matrix[k][j] == 1:
                ref_matrix[k] = (ref_matrix[k] + ref_matrix[i]) % 2
    return ref_matrix


def get_lead_indexes(matrix):
    """Получение индексов ведущих столбцов"""
    ref_matrix = ref(matrix)
    row = ref_matrix.shape[0]
    return [np.nonzero(ref_matrix[i] == 1)[0][0] for i in range(row)]


def reduce_matrix(matrix):
    """Формирование сокращённой матрицы"""
    edited_matr = rref(matrix)
    lead_idxs = get_lead_indexes(matrix)
    return np.delete(edited_matr, lead_idxs, axis=1)


def join_matrix(matrix):
    """Формирование проверочной матрицы H"""
    X = reduce_matrix(matrix)
    I = np.eye(X.shape[1], dtype=int)
    result_matrix = np.zeros((X.shape[0] + X.shape[1], X.shape[1]), dtype=int)
    lead_idxs = get_lead_indexes(matrix)
    x_idx, i_idx = 0, 0
    for idx in range(result_matrix.shape[0]):
        if idx in lead_idxs:
            result_matrix[idx] = X[x_idx]
            x_idx += 1
        else:
            result_matrix[idx] = I[i_idx]
            i_idx += 1
    return result_matrix


def generate_code_words(matrix):
    """Формирование кодовых слов длины n"""
    row, col = matrix.shape
    words = []
    for i in range(1, 2 ** row):
        bin_word = [int(x) for x in format(i, f'0{row}b')]
        word = np.dot(bin_word, matrix) % 2
        words.append(word)
    unique_words = np.unique(words, axis=0)
    return unique_words


def check_code_word(matrix, H):
    """Проверка умножения кодовых слов на H"""
    code_words = generate_code_words(matrix)
    for word in code_words:
        if not np.all(np.dot(word, H) % 2 == 0):
            return False
    return True


def code_distance(matrix):
    """Вычисление кодового расстояния"""
    code_words = generate_code_words(matrix)
    min_weight = min([np.sum(word) for word in code_words if np.sum(word) > 0])
    return min_weight


def error_detection(matrix, H, t):
    """Внесение ошибки кратности t и проверка обнаружения"""
    code_words = generate_code_words(matrix)
    word = code_words[0]
    error_vector = np.zeros(word.shape, dtype=int)
    error_vector[:t] = 1  # Вносим ошибку в t первых битов
    error_word = (word + error_vector) % 2
    return np.dot(error_word, H) % 2  # Должно вернуть ненулевой вектор


class LinearCode:
    def __init__(self, matrix):
        self.matrix = matrix
        self.S_REF = ref(self.matrix)
        self.G_star = rref(self.S_REF)
        self.lead = get_lead_indexes(self.G_star)
        self.X = reduce_matrix(self.G_star)
        self.H = join_matrix(self.matrix)
        self.run_all_tasks()

    def run_all_tasks(self):
        """Выполнение всех заданий и вывод результатов в консоль"""

        # Задание 1.3.1
        print("1.3.1. Приведение матрицы к ступенчатому виду (REF):\n", self.S_REF)

        # Задание 1.3.2
        print("1.3.2. Приведение матрицы к приведённому ступенчатому виду (RREF):\n", self.G_star)
        n, k = self.G_star.shape[1], self.G_star.shape[0]
        print(f"Число столбцов (n) = {n}, Число строк (k) = {k}")

        # Задание 1.3.3
        print("1.3.3. Проверочная матрица H:\n", self.H)

        # Задание 1.4.1
        code_words = generate_code_words(self.G_star)
        print("1.4.1. Кодовые слова длины n:\n", code_words)

        # Задание 1.4.2
        check = check_code_word(self.G_star, self.H)
        print(f"1.4.2. Результат проверки кодовых слов (все нули): {'Да' if check else 'Нет'}")

        # Задание 1.5
        distance = code_distance(self.G_star)
        print(f"1.5. Кодовое расстояние: {distance}")

# Создание экземпляра класса
linear_code = LinearCode(S)
