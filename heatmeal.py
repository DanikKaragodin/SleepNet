import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Матрица путаницы
confusion_matrix = np.array([
    [9135, 745, 125, 35, 157],
    [491, 1464, 576, 12, 261],
    [421, 750, 15383, 543, 702],
    [47, 29, 784, 4819, 24],
    [223, 627, 803, 4, 6060]
])

# Метки классов
class_labels = ['Wake', 'N1', 'N2', 'N3', 'REM']

# Создание тепловой карты
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Greens', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Предсказанное')
plt.ylabel('Истинное')
plt.title('Матрица путаницы')
plt.show()
