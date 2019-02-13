import keras  # Keras 2.1.2 and TF-GPU 1.9.0
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import TensorBoard
import numpy as np
import os
import random
import tensorflow as tf

train_data_dir = "train_data"
training_data = []

all_files = os.listdir(train_data_dir)

for file in all_files:
    full_path = os.path.join(train_data_dir, file)
    data = np.load(full_path)
    data = data[0]
    print(data)

    BO = data[0]
    data = np.delete(data, 0, 0)
    print(data)
    training_data.append([data, BO])

# print('Training data:', training_data)
random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    if label == 99:
        continue
    else:
        X.append(features)
        y.append(label)

# print('X:', X)
# print('y:', y)


X = np.array(X).reshape(-1, 54, 1)
# print('X reshape:', X)
y = np.array(y).reshape(-1)
# print('y reshape:', y)
model = Sequential()

model.add(Flatten())
model.add(Dense(150, activation=tf.nn.relu, input_dim=54))  # a simple fully-connected layer, 150 units, relu activation
model.add(Dropout(0.2))
model.add(Dense(300, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(600, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(300, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(150, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(
    Dense(5, activation=tf.nn.softmax)
)  # our output layer. 5 units for 5 Build orders. Softmax for probability distribution

learning_rate = 0.001
opt = keras.optimizers.adam(lr=learning_rate, decay=1e-6)

model.compile(
    optimizer=opt,  # Good default optimizer to start with
    loss="sparse_categorical_crossentropy",  # how will we calculate our "error." Neural network aims to minimize loss.
    metrics=["accuracy"],
)  # what to track
tensorboard = TensorBoard(log_dir="logs/STAGE1")
# Train the model
model.fit(X, y, epochs=10000, callbacks=[tensorboard])

model.save("MadAI_09_02_2019")

# print(X[0])
x_test = X
prediction = model.predict(x_test)
# print(prediction)
# print(np.argmax(prediction[0]))

# val_loss, val_acc = model.evaluate(x_test, y_test)  # evaluate the out of sample data with model
# print(val_loss)  # model's loss (error)
# print(val_acc)  # model's accuracy
