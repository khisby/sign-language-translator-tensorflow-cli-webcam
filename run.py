import tensorflow as tf
import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class_indices = np.load('model/index_class.npy', allow_pickle=True).item()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation=tf.keras.activations.relu, padding = 'Same', strides=1, use_bias=True, kernel_initializer=tf.keras.initializers.glorot_normal(), bias_initializer=tf.keras.initializers.zeros(), data_format="channels_last", input_shape=(28, 28, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2),  data_format="channels_last"),

    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.keras.activations.relu, padding='Same', strides=1, kernel_initializer=tf.keras.initializers.VarianceScaling(),  data_format="channels_last"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2),  data_format="channels_last"),

    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation=tf.keras.activations.relu, padding='Same', strides=1, kernel_initializer=tf.keras.initializers.VarianceScaling(),  data_format="channels_last"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2),  data_format="channels_last"),

    tf.keras.layers.Flatten(data_format="channels_last"),
    tf.keras.layers.Dense(512, activation=tf.keras.activations.relu, kernel_initializer=tf.keras.initializers.VarianceScaling()),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(class_indices), activation=tf.keras.activations.softmax, kernel_initializer=tf.keras.initializers.VarianceScaling)
])

model.load_weights("model/model.h5")

camera = cv2.VideoCapture(0)
x, y, w, h = 350, 150, 200, 200

teks = ""
arr = []
while True:
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Caputer a image", frame)

    gambar_crop = frame[y:y+h, x:x+w]
    gambar_crop = cv2.cvtColor(gambar_crop, cv2.COLOR_BGR2GRAY)

    gambar_crop = cv2.resize(gambar_crop, (28, 28))
    img_predict = np.expand_dims(img_to_array(gambar_crop), axis=0)
    datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = datagen.flow(img_predict, batch_size=1)
    predict = model.predict(train_generator)
    predict_arg_max = predict.argmax(axis=-1)
    print(predict)

    if(predict[0][predict_arg_max[0]] == 1.0):
        for name, age in class_indices.items():
            if age == predict_arg_max:
                arr.append(name)

                if(len(arr) >= 20):
                    if(len(set(arr)) == 1):
                        teks += arr[0]
                        arr.clear()
                        print(teks)
                    arr.clear()

    keypress = cv2.waitKey(1)

    if keypress == 27:
        break

camera.release()
cv2.destroyAllWindows()
