import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization

# Loading data separated with comma
dataset = np.loadtxt('glass.csv', delimiter=',')

# Permutating data
np.take(dataset, np.random.permutation(dataset.shape[0]), axis=0, out=dataset)

# Dividing data into inputs and outputs
X = dataset[:, 1:10]
y = dataset[:, 10]

# Formatting output data
formatted_y = []
for number in y:
    array = np.zeros(int(max(y)))
    array[int(number)-1] = 1
    formatted_y.append(array)
formatted_y = np.array(formatted_y)

# Dividing data into training (80%) and testing (20%)
train_x = X[:171, :]
train_y = formatted_y[:171, :]
test_x = X[171:, :]
test_y = formatted_y[171:, :]

# Creating model
model = Sequential()
model.add(BatchNormalization())
model.add(Dense(30, input_dim=9, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(7, activation='softmax'))

# Configuring model
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Network learning
model.fit(train_x, train_y, epochs=1000, batch_size=32,
          verbose=0, use_multiprocessing=True)

# Showing the results
_, test_accuracy = model.evaluate(test_x, test_y)
_, train_accuracy = model.evaluate(train_x, train_y)
print('Train accuracy: %.2f' % (train_accuracy*100))
print('Test accuracy: %.2f' % (test_accuracy*100))

# Prediction on test data
results = model.predict(test_x)
formattedNetworkResults = []
formattedResults = []

# Neural output
for result in results:
    formattedNetworkResults.append(result.tolist().index(max(result)) + 1)

# Expected output
for result in test_y:
    formattedResults.append(result.tolist().index(max(result)) + 1)

# Showing results on plot
plt.figure(figsize=(15, 8))
plt.title("Test data - accuracy: %.2f %%" % (test_accuracy*100))
plt.plot(formattedNetworkResults, 'ro', label="Expected output")
plt.plot(formattedResults, 'go', label="Network output")
plt.legend(loc="best")
plt.show()
