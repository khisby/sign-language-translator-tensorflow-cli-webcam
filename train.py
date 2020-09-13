import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if os.path.exists('data'):
    shutil.rmtree("data/", ignore_errors=False, onerror=None)

if os.path.exists('model'):
    shutil.rmtree("model/", ignore_errors=False, onerror=None)

os.mkdir('data/')
os.mkdir('model/')
os.mkdir('data/train/')
os.mkdir('data/test/')

for i in os.listdir("dataset"):
        jumlah_file = len([name for name in os.listdir("dataset/" + i) if os.path.isfile(os.path.join("dataset/" + i, name))])
        persen = round(jumlah_file * 0.2)
        print("Kelas : " + str(i) + ", jumlah : " + str(jumlah_file) + ", (train, test) : ({},{})".format(jumlah_file-persen, persen))

        if not os.path.exists("data/train/" + str(i)):
            os.mkdir("data/train/" + str(i))
        if not os.path.exists("data/test/" + str(i)):
            os.mkdir("data/test/" + str(i))


        for index,berkas in enumerate(os.listdir("dataset/"+i)):
            if index < persen:
                shutil.copyfile("dataset/{}/".format(i) + str(berkas),
                          "data/train/" + i + "/" + str(berkas))
            else:
                shutil.copyfile("dataset/{}/".format(i) + str(berkas),
                          "data/test/" + i + "/" + str(berkas))


root_dir = 'data/'
data_train = os.path.join(root_dir, 'train/')
data_test = os.path.join(root_dir, 'test/')

jumlah_file_dalam_folder = len([name for name in os.listdir("dataset") if os.path.isfile(os.path.join("dataset", name))])

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(data_train, target_size=(28, 28), batch_size=2, class_mode='categorical', color_mode='grayscale')
test_generator = test_datagen.flow_from_directory(data_test, target_size=(28, 28), batch_size=2, class_mode='categorical', color_mode='grayscale')

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
    tf.keras.layers.Dense(len(train_generator.class_indices), activation=tf.keras.activations.softmax, kernel_initializer=tf.keras.initializers.VarianceScaling)
])

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])
# save_model = tf.keras.callbacks.ModelCheckpoint(filepath="model/cp.ckpt", save_weights_only=True, verbose=1)
model.fit(train_generator, epochs=100, validation_data=test_generator, verbose=1, callbacks=[])

model.save_weights('model/model.h5', overwrite=True)
np.save('model/index_class', train_generator.class_indices)
