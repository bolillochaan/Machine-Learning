import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Cargardo el MNIST
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")

# Seleccionaremos 10k de muestras aleatorias
np.random.seed(42)
indices = np.random.choice(len(mnist.data), 10000, replace=False)
X = mnist.data[indices]
y = mnist.target[indices].astype(int)  # Hay que Convertir las  etiquetas a enteros


# Ejemplos del entrenamiento
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, X[:4], y[:4]):
    ax.set_axis_off()
    ax.imshow(image.reshape(28, 28), cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Training: {label}")

# Creando el clasificador
clf = svm.SVC(gamma="scale", C=10, kernel="rbf")  # Configuración para MNIST

# Dividir los datos en este caso seran el 80% Entrenamiento y 20% Prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42
)

# Entrenamos
clf.fit(X_train, y_train)

# Predecimos
predicted = clf.predict(X_test)

# Metricas
print("Classification report:\n", metrics.classification_report(y_test, predicted))
report = metrics.classification_report(y_test, predicted, output_dict=True)
print("Confusion matrix:\n", metrics.confusion_matrix(y_test, predicted))

# Predicciones con etiquetas reales
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction, true_label in zip(axes, X_test[:4], predicted[:4], y_test[:4]):
    ax.set_axis_off()
    ax.imshow(image.reshape(28, 28), cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Pred: {prediction}\nTrue: {true_label}")

#Grafica de las Metricas
df_report = pd.DataFrame(report).transpose().drop("accuracy", errors="ignore")
df_classes = df_report[df_report.index.astype(str).str.isdigit()]

fig, ax = plt.subplots(figsize=(12, 6))
df_classes[["precision", "recall", "f1-score"]].plot(kind="bar", ax=ax)
plt.title("Métricas por Clase")
plt.xticks(rotation=0)
plt.ylim(0, 1.1)
plt.legend(loc="lower right")
plt.grid(axis="y", linestyle="--")

for i, (idx, row) in enumerate(df_classes.iterrows()):
    ax.text(i, -0.15, f"Sup: {int(row['support'])}", ha="center", fontsize=8)

plt.tight_layout()
#Grafica de la Matriz de Confusion
classes = np.unique(np.concatenate([y_test, predicted]))
disp = metrics.ConfusionMatrixDisplay.from_predictions(
    y_test,
    predicted,
    display_labels=classes,
    cmap=plt.cm.Blues,
    colorbar=False,
    values_format="d"
)
disp.ax_.set_title("Matriz de Confusión")

plt.show()
