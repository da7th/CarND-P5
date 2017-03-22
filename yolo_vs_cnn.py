#imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
import glob
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from skimage.feature import hog
import imghdr
from scipy.ndimage.measurements import label


img_full = mpimg.imread('/home/da7th/Desktop/Udacity/sdc/CarND-P5/test_images/test1.jpg')
img_car = mpimg.imread('/home/da7th/Desktop/Udacity/sdc/CarND-P5/vehicles/GTI_Right/image0301.png')
img_notcar = mpimg.imread('/home/da7th/Desktop/Udacity/sdc/CarND-P5/non-vehicles/Extras/extra16.png')



def load_image_sets():

    #car folders:
    imgs_png1 = glob.glob('/home/da7th/Desktop/Udacity/sdc/CarND-P5/vehicles/GTI_Far/*.png')
    imgs_png2 = glob.glob('/home/da7th/Desktop/Udacity/sdc/CarND-P5/vehicles/GTI_Left/*.png')
    imgs_png3 = glob.glob('/home/da7th/Desktop/Udacity/sdc/CarND-P5/vehicles/GTI_MiddleClose/*.png')
    imgs_png4 = glob.glob('/home/da7th/Desktop/Udacity/sdc/CarND-P5/vehicles/GTI_Right/*.png')
    imgs_png5 = glob.glob('/home/da7th/Desktop/Udacity/sdc/CarND-P5/vehicles/KITTI_extracted/*.png')

    #notcar folders:
    imgs_png6 = glob.glob('/home/da7th/Desktop/Udacity/sdc/CarND-P5/non-vehicles/Extras/*.png')
    imgs_png7 = glob.glob('/home/da7th/Desktop/Udacity/sdc/CarND-P5/non-vehicles/GTI/*.png')

    #concatenate all the similar images
    car_imgs = np.concatenate((imgs_png1,imgs_png2,imgs_png3,imgs_png4,imgs_png5))
    notcar_imgs = np.concatenate((imgs_png6,imgs_png7))

    return car_imgs, notcar_imgs

car_imgs, notcar_imgs = load_image_sets()
print("The number of non-car images is: ",len(notcar_imgs))
print("The number of car images is: ",len(car_imgs))


def equalise_sets(car_features, notcar_features):
    print("Set lengths before:", len(car_features),len(notcar_features))

    car_features = shuffle(car_features)
    notcar_features = shuffle(notcar_features)

    if len(car_features) > len(notcar_features):
        car_features = car_features[:len(notcar_features)]
    elif len(car_features) < len(notcar_features):
        notcar_features = notcar_features[:len(car_features)]

    print("Set lengths after equalise function:", len(car_features),len(notcar_features))

    return car_features, notcar_features


import tensorflow as tf
tf.python.control_flow_ops = tf

print('Modules loaded.')
car_images = []
notcar_images = []
images = []
size = (224,224)
for fname in car_imgs[:10]:
    image = cv2.resize((mpimg.imread(fname)), size)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    car_images.append(image)
for fname in notcar_imgs[:10]:
    image = cv2.resize((mpimg.imread(fname)), size)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    notcar_images.append(image)
print(len(car_images),len(notcar_images))
print(car_images[0].shape,notcar_images[0].shape)


def more_data(images):
    print(images[0].shape)
    all_images = []
    for image in images:
        int_rand = np.random.randint(4,24)
#         print(int_rand)
        all_images.append(cv2.resize(image[int_rand:,int_rand:], size))


    print(all_images[0].shape)
    return np.concatenate((all_images,images),axis=0)
# more_data([img_car])
# car_image = cv2.resize(img_car, size)



def cnn_train_2(car_images, notcar_images):
#     print(car_features[0].shape)
    car_features, notcar_features = equalise_sets(car_images, notcar_images)
#     car_features = more_data(car_features)
#     notcar_features = more_data(notcar_features)

    print("Set lengths after augmentation function:", len(car_features),len(notcar_features))

    X = np.vstack((notcar_features[:], car_features[:])).astype(np.float64)
    # Define the labels vector
    print(X.shape)
    y = np.hstack((np.ones(len(car_features[:])), np.zeros(len(notcar_features[:]))))
    print(y.shape)
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)

    X_train, y_train = shuffle(X_train, y_train)
    from sklearn.preprocessing import MinMaxScaler
    X_normalized = np.subtract(X_train/X_train.max(axis=0),0.5)

    X_test_normalized = X_test/X_test.max(axis=0) -0.5

    print("X_normalized: ", X_normalized.shape)
    with tf.Session() as sess:

        y_one_hot = sess.run(tf.one_hot(y_train, 2))
        y_test_one_hot = sess.run(tf.one_hot(y_test, 2))


    from keras.models import Sequential
    from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
    from keras.layers.core import Dense, Activation, Flatten, Dropout
    from keras.layers.advanced_activations import LeakyReLU

    model = Sequential()

    model.add(Convolution2D(4, 3, 3, border_mode='same', input_shape=(224,224,3)))

    model.add(LeakyReLU(alpha=0.3))

    model.add(MaxPooling2D(pool_size = (2,2)))


    model.add(Convolution2D(8, 3, 3))

    model.add(LeakyReLU(alpha=0.3))

    model.add(MaxPooling2D(pool_size = (2,2)))



    model.add(Convolution2D(16, 3, 3))

    model.add(LeakyReLU(alpha=0.3))

    model.add(MaxPooling2D(pool_size = (2,2)))



    model.add(Convolution2D(32, 3, 3))

    model.add(LeakyReLU(alpha=0.3))

    model.add(MaxPooling2D(pool_size = (2,2)))



    model.add(Convolution2D(64, 3, 3))

    model.add(LeakyReLU(alpha=0.3))

    model.add(MaxPooling2D(pool_size = (2,2)))



    model.add(Convolution2D(128, 3, 3))

    model.add(LeakyReLU(alpha=0.3))

    model.add(MaxPooling2D(pool_size = (2,2)))



    model.add(Convolution2D(256, 3, 3))

    model.add(LeakyReLU(alpha=0.3))

    model.add(MaxPooling2D(pool_size = (2,2)))



    model.add(Flatten())

    model.add(Dense(1470))

    model.add(Activation('linear'))

    model.add(Dense(2))

    model.add(Activation('softmax'))

    # Configures the learning process and metrics
    model.compile('adam', 'categorical_crossentropy', ['accuracy'])

    # Train the model
    # History is a record of training loss and metrics
    history = model.fit(X_train, y_one_hot, batch_size=32, nb_epoch=30, validation_split=0.2)

    # Calculate test score
    test_score = model.evaluate(X_test, y_test_one_hot)
    print("test score:", test_score)
    print('Test score:', test_score[0])
    print('Test accuracy:', test_score[1])

    model.save("model_yolo3.h5")

#     n_predict = 10
#     print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
#     print('For these',n_predict, 'labels: ', y_test[0:n_predict])

    return model

# print(svc.dtype)








































yolo = cnn_train_2(car_images,notcar_images)

# cnn = cnn_train(car_images,notcar_images)
