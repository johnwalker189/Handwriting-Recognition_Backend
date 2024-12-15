import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train_knn = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test_knn = X_test.reshape(X_test.shape[0], -1) / 255.0

def knn_predict(X_train, y_train, X_test, k=3):
    predictions = []
    for test_point in X_test:
       
        distances = np.linalg.norm(X_train - test_point, axis=1)
        
        k_indices = distances.argsort()[:k]
       
        k_labels = y_train[k_indices]
        
        unique, counts = np.unique(k_labels, return_counts=True)
        predictions.append(unique[np.argmax(counts)])
    return np.array(predictions)

knn_predictions = knn_predict(X_train_knn, y_train, X_test_knn[:100], k=3)
knn_accuracy = np.mean(knn_predictions == y_test[:100])
print({knn_accuracy * 100:.2f}%")
