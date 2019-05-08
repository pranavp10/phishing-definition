from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)
# load  dataset
dataset = numpy.loadtxt("datasets/phishcoop.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,:-1]
Y = dataset[:,-1]

# create model
model = Sequential()
model.add(Dense(12, input_dim=22, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=150, batch_size=10)

scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
model.save("model.pkl")