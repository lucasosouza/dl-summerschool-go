
# Criando sua primeira MLP em Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy

# fix random seed for reproducibility
numpy.random.seed(7)

# Carregando pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.data", delimiter=",")

size = 600

# definindo  input (X) e output (Y) 
X = dataset[:size,:8]
Y = dataset[:size,8]

X_val = dataset[size:,:8]
Y_val = dataset[size:,8]


# criando modelo
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compilando Modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treinando
model.fit(X, Y, epochs=30, batch_size=10, validation_data=[X_val, Y_val])

model.fit(X, Y, epochs=30, batch_size=10, validation_split=0.2)

# Avaliando
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))