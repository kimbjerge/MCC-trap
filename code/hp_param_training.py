# -*- coding: utf-8 -*-
"""
Created on Sun May 24 12:19:05 2020

@author: Kim Bjerge
"""


# %Reference - https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams
# Last accessed 18/12/19
#https://stackoverflow.com/questions/38543850/tensorflow-how-to-display-custom-images-in-tensorboard-e-g-matplotlib-plots
# Last accessed 18/12/19

# activate moths
# Anaconda: 
#    conda install scikit-learn
#    conda install seaborn

#%load_ext tensorboard
import tensorflow as tf
import io

## LIMIT MEMORY - Can be uncommented
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.compat.v1.Session(config=config)

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sklearn

from tensorboard.plugins.hparams import api as hp

print(tf.__version__)
print(sklearn.__version__)

##

# Directory with subdirectories for each class with cropped images in jpg format
#data_dir = '/home/don/moths/10classes_mixed'
data_dir = 'H:/Natsvaermer/TrainingData2020/10classes_mixed'

# Directory for saving tensorboard model parameters
hparam_dir = '..\\hparam_tuning' # Windows path

# Directory for saving h5 models for each run
models_dir = '../models_save'

gen = True # Enable data augmentation

##

number_of_classes = 10
batch_size = 32
epochs = 30
image_size = 128
seed = 1

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
    rotation_range = 180,
    horizontal_flip = True,
    vertical_flip = True,
    zoom_range=0.3,
    validation_split=0.2,
    brightness_range = [0.9, 1.1])

train_generator = train_datagen.flow_from_directory(
    data_dir,
    shuffle = True,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    seed = seed)

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)    
validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=seed
)

##

# Selected best model finnaly used (Paper rating no. 3)
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam']))
HP_KERNEL_SIZE1 = hp.HParam('kern_size1', hp.Discrete([5]))
HP_KERNEL_SIZE2 = hp.HParam('kern_size2', hp.Discrete([3]))
HP_NUM_FILTERS1 = hp.HParam('num_filters1', hp.Discrete([32])) 
HP_NUM_FILTERS2 = hp.HParam('num_filters2', hp.Discrete([64])) 
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([512]))

# Hyperparameter tuning used to find optimal model
#HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam','sgd']))
#HP_KERNEL_SIZE1 = hp.HParam('kern_size1', hp.Discrete([3,5]))
#HP_KERNEL_SIZE2 = hp.HParam('kern_size2', hp.Discrete([1,3]))
#HP_NUM_FILTERS1 = hp.HParam('num_filters1', hp.Discrete([32, 64])) 
#HP_NUM_FILTERS2 = hp.HParam('num_filters2', hp.Discrete([64, 128])) 
#HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([256, 512]))

METRIC_ACCURACY = 'accuracy'
METRIC_F1 = 'f1'
    
with tf.summary.create_file_writer(hparam_dir).as_default():
      hp.hparams_config(
        hparams=[HP_NUM_UNITS,
                 HP_OPTIMIZER,
                 HP_NUM_FILTERS1,
                 HP_NUM_FILTERS2,
                 HP_KERNEL_SIZE1,
                 HP_KERNEL_SIZE2],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy'), hp.Metric(METRIC_F1, display_name='F1 Score')],
          )
      
##

def train_test_model(logdir, hparams, count):  
    model = tf.keras.models.Sequential()
    #Block 1
    model.add(tf.keras.layers.Conv2D(hparams[HP_NUM_FILTERS1], kernel_size=hparams[HP_KERNEL_SIZE1], input_shape=(image_size,image_size, 3), padding='same'))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    #Block 2 
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), input_shape=(image_size,image_size, 3),padding='same'))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    #Block 3 
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), input_shape=(image_size,image_size, 3),padding='same'))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    #Block 4
    model.add(tf.keras.layers.Conv2D(hparams[HP_NUM_FILTERS2], hparams[HP_KERNEL_SIZE2], input_shape=(image_size,image_size, 3),padding='same'))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    #Dense
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    #Output
    model.add(tf.keras.layers.Dense(number_of_classes, activation='softmax')) # NUM CLASSES
    #model.add(tf.keras.layers.Dense(10,activation='sigmoid')) # NUM CLASSES
    model.compile(optimizer=hparams[HP_OPTIMIZER],
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    print("Learnable parameters:", model.count_params())
    
    if not gen:
        model.fit(x_train, y_train, 
                  epochs = epochs,
                  batch_size = batch_size,
                  callbacks=[tf.keras.callbacks.TensorBoard(logdir),#log metrics 
                            hp.KerasCallback(logdir, hparams),      #log hparams
                            ],)
        _, accuracy = model.evaluate(x_test, y_test)
        y_predicted = model.predict(x_test)
        y_pred = np.argmax(y_predicted, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        print(y_pred.shape)
        print(y_true.shape)
        report = classification_report(y_true, y_pred, output_dict=True)
        print(classification_report(y_true, y_pred))
        f1_score = report['weighted avg']['f1-score']
        return accuracy, f1_score, y_pred
    else:
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
        rotation_range = 180,
        horizontal_flip = True,
        vertical_flip = True,
        zoom_range=0.3,
        validation_split=0.2,
        brightness_range = [0.9, 1.1])

        train_generator = train_datagen.flow_from_directory(
            data_dir,
            shuffle = True,
            target_size=(image_size, image_size),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            seed = seed)

        validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)    
        validation_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(image_size, image_size),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False,
            seed=seed
        )
        print('Model train using generator')
        model.fit_generator(train_generator,
                            steps_per_epoch=(1984//batch_size)*4,
                            epochs=epochs,
                            callbacks=[
                                        tf.keras.callbacks.TensorBoard(logdir),  # log metrics
                                        hp.KerasCallback(logdir, hparams),       # log hparams
                                      ],)
        print('Model evaluate')
        _, accuracy = model.evaluate_generator(validation_generator)
        print('Model predict')
        Y_pred = model.predict_generator(validation_generator) #, 173//batch_size+1
        y_pred = np.argmax(Y_pred, axis=1)
        print('Confusion Matrix')
        print(classification_report(validation_generator.classes, y_pred))
        report = classification_report(validation_generator.classes, y_pred, output_dict=True)
        f1_score = report['weighted avg']['f1-score']
        model.save(models_dir + '/' + str(count) + '.h5')
        return accuracy, f1_score, y_pred

##
        
#https://androidkt.com/keras-confusion-matrix-in-tensorboard/
def run(run_dir, hparams, count):
    if not gen:
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            accuracy, f1_score, y_pred = train_test_model(run_dir, hparams)
            tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)
            tf.summary.scalar(METRIC_F1, f1_score, step=1)
            conf = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
            figure = plt.figure(figsize=(8, 8))
            sns.heatmap(conf, annot=True,cmap=plt.cm.Blues)
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close(figure)
            buf.seek(0)
            image = tf.image.decode_png(buf.getvalue(), channels=4)
            image = tf.expand_dims(image, 0)
            tf.summary.image("Confusion Matrix/scoref1:_" + str(f1_score), image, step=1)
    else:
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            accuracy, f1_score, y_pred = train_test_model(run_dir, hparams, count)
            tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)
            tf.summary.scalar(METRIC_F1, f1_score, step=1)
            conf = confusion_matrix(validation_generator.classes, y_pred)
            figure = plt.figure(figsize=(8, 8))
            sns.heatmap(conf, annot=True,cmap=plt.cm.Blues)
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close(figure)
            buf.seek(0)
            image = tf.image.decode_png(buf.getvalue(), channels=4)
            image = tf.expand_dims(image, 0)
            tf.summary.image("Confusion Matrix/scoref1:_" + str(f1_score) + '_run_' + str(count), image, step=1)
            
##
            
session_num = 0
for num_units in HP_NUM_UNITS.domain.values:
    for optimizer in HP_OPTIMIZER.domain.values:
        for kern_size1 in HP_KERNEL_SIZE1.domain.values:
            for kern_size2 in HP_KERNEL_SIZE2.domain.values:
                for num_filters1 in HP_NUM_FILTERS1.domain.values:
                    for num_filters2 in HP_NUM_FILTERS2.domain.values:
                        hparams = {
                          HP_NUM_UNITS: num_units,
                          HP_OPTIMIZER: optimizer,
                          HP_KERNEL_SIZE1: kern_size1,
                          HP_KERNEL_SIZE2: kern_size2,
                          HP_NUM_FILTERS1: num_filters1,
                          HP_NUM_FILTERS2: num_filters2
                        }
                        run_name = "run-%d" % session_num
                        print('--- Starting trial: %s' % run_name)
                        print({h.name: hparams[h] for h in hparams})
                        run(hparam_dir + '\\' + run_name, hparams, session_num) # Windows path
                        session_num += 1
                        
#model.summary()
