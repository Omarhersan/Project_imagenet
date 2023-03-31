import os
import random
import cv2 as cv
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ConvNeXtXLarge
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model

class DataPreparator:

    def __init__(self, train_path, val_path, test_path, batch_size = 32) -> None:
        self.train_path = train_path
        self.test_path  = test_path
        self.val_path   = val_path
        self.batch_size = batch_size
        self.trainData  = None
        self.testData   = None
        self.valData    = None
        pass

    def preprocess_train(self, img_src, annotation, numClasses):
        annotation = annotation.rstrip('\n')
        filename, x_min, y_min, x_max, y_max = annotation.split("\t")
        img_class, _ = filename.split('_')
        img_class = str(numClasses[img_class])
        img_path = img_src + "/" + filename
        img = cv.imread(str(img_path))
        h,w, _ = img.shape
        annotation = "".join([img_path, ",",
                            img_class, ',',
                            str(float(x_min)/h),",",
                            str(float(y_min)/w),",",
                            str(float(x_max)/h),",",
                            str(float(y_max)/w)
                            ])
        return annotation

    def build_train(self):
        classes = os.listdir(self.train_path)
        numericalClasses = {j: i for i, j in enumerate(classes)}
        train_data = []
        for clas in classes:
            iPath = self.train_path + clas + '/images'
            aPath = self.train_path + clas + f'/{clas}_boxes.txt'
            annotations = open(aPath).read().splitlines()
            examples = [self.preprocess_train(iPath, item, numericalClasses) for item in annotations]
            train_data+=examples

        return train_data

    #Este sirve para train y val
    def loadExample(self, example):
        str_tensors = tf.strings.split(example, sep = ',')

        #Cargar imagen
        img = tf.io.read_file(str_tensors[0])
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img = tf.image.resize(img, (50, 50))

        img_Class = tf.strings.to_number(str_tensors[1])

        x_min = tf.strings.to_number(str_tensors[2])
        y_min = tf.strings.to_number(str_tensors[3])
        x_max = tf.strings.to_number(str_tensors[4])
        y_max = tf.strings.to_number(str_tensors[5])

        bbox = (x_min, y_min, x_max, y_max)

        return (img, img_Class, bbox)
    
    def buildTrainDataset(self):
        train_data = self.build_train()
        train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
        train_dataset = (train_dataset
                        .shuffle(len(train_data))
                        .map(self.loadExample, num_parallel_calls = tf.data.AUTOTUNE)
                        .cache().batch(self.batch_size)
                        .prefetch(tf.data.AUTOTUNE)
                        )
        
        self.trainData = train_dataset
        
    def preprocess_val(self, img_src, annotation):
        annotation = annotation.rstrip('\n')
        filename, img_class, x_min, y_min, x_max, y_max = annotation.split("\t")
        img_path = img_src + "/" + filename
        img = cv.imread(str(img_path))
        h,w, _ = img.shape
        annotation = "".join([img_path, ",",
                            img_class, ',',
                            str(float(x_min)/h),",",
                            str(float(y_min)/w),",",
                            str(float(x_max)/h),",",
                            str(float(y_max)/w)
                            ])
        return annotation

    def build_val(self):
        iPath = self.val_path + '/images'
        aPath = self.val_path + '/val_annotations.txt'
        annotations = open(aPath).read().splitlines()
        examples = [self.preprocess_val(iPath, item) for item in annotations]

        return examples
    
    def buildValDataset(self):
        val_data = self.build_val()
        val_dataset = tf.data.Dataset.from_tensor_slices(val_data)
        val_dataset = (val_dataset
                        .shuffle(len(val_data))
                        .map(self.loadExample, num_parallel_calls = tf.data.AUTOTUNE)
                        .cache().batch(self.batch_size)
                        .prefetch(tf.data.AUTOTUNE)
                        )
        
        self.valData = val_dataset

    def build_test(self):
        iPath = self.test_path + '/images'
        img = os.listdir(iPath)
        examples = [f'{iPath}/{item}' for item in img]

        return examples

    def loadTest(self, example):
        str_tensors = tf.strings.split(example, sep = ',')

        #Cargar imagen
        img = tf.io.read_file(str_tensors[0])
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img = tf.image.resize(img, (50, 50))

        return (img)

    def buildTestDataset(self):
        test_data = self.build_test()
        test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
        test_dataset = (test_dataset
                    .shuffle(len(test_data))
                    .map(self.loadTest, num_parallel_calls = tf.data.AUTOTUNE)
                    .cache().batch(32)
                    .prefetch(tf.data.AUTOTUNE)
                    )
        
        self.testData = test_dataset

    #Solo se tiene que mandar llamar esta función para obtener los tres tensores
    def returnData(self):
        self.buildTrainDataset()
        self.buildValDataset()
        self.buildTestDataset()

        return self.trainData, self.valData, self.testData