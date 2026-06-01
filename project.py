from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 1. Завантаження датасету (Оптичне розпізнавання цифр)
digits = load_digits()
X = digits.data      # Ознаки: пікселі зображень (масиви 8x8, розгорнуті в рядок з 64 елементів)
y = digits.target    # Класи: правильні відповіді (цифри від 0 до 9)

# 2. Розбиття на тренувальну (80%) та тестову (20%) вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Вибір та навчання моделі
# Використовуємо метод k-найближчих сусідів (k-NN) - один з найпростіших алгоритмів
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# 4. Прогнозування та оцінка результатів
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Точність моделі (Accuracy): {accuracy * 100:.2f}%\n")
print("Детальний звіт по класах:")
print(classification_report(y_test, y_pred))

# 5. Візуалізація результатів (вивід кількох тестових зображень та прогнозів моделі)
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for ax, image, prediction, actual in zip(axes.flat, X_test, y_pred, y_test):
    ax.set_axis_off()
    # Повертаємо формат зображення 8x8 для відображення
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Прогноз: {prediction}\nФакт: {actual}")

plt.tight_layout()
plt.savefig('result_plot.png')