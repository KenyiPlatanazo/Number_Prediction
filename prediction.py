from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

model = load_model("model_number_prediction.h5")

test_data = pd.read_csv("./Train.csv")
X_test = test_data.iloc[:, 1:].values / 255.0
X_test = X_test.reshape(-1, 28, 28, 1)

predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

indices = np.random.choice(len(X_test),5, replace=False)
for i in indices:
    plt.imshow(X_test[i].reshape(28,28), cmap="gray")
    plt.title(f"Predicted: {predicted_labels[i]}")
    plt.axis("off")
    plt.show()
