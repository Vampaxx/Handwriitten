from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,MaxPooling2D,Flatten


def model_():
    model= Sequential()

    model.add(Conv2D(32, (3,3), input_shape=(28,28,1)))
    model.add(MaxPooling2D())

    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D())


    model.add(Flatten())
    model.add(Dense(10, activation= 'relu'))
    model.add(Dense(10, activation= 'softmax'))
    return model
