#Plant Diseases Detection


# Part 1 - Building the CNN
 
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
"""(We create many feature maps using many feature detectors to obtain our 1st convolutional layer).After execute,1st convolutional layer is added.

Convolution2D(No of feature detector,its row,its col,border_mode='same' meaning how to feature detector handle the border of image but no need,input_shape means shape of input image for theano backend(3,64,64)->3 channel means colored array||dimension of 2d array in  each channel-gpu use so larger format like 128 by 128 or 256 by 256,activation='relu' for nonlinearity cause classifying images is non linear problem so rectifier function is relu)"""

classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
"""Here,from feature map we pick up highest number by sliding a 2*2 sub table to right.By doing it,feature map reduces its size called Pooled feature map.
MaxPooling2D(pool_size = (2, 2)-sub table which size 2 by 2(most of the time)"""

classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
"""Improve model-1.Add another convolutional layer(best soln) 2.Add another fully conected layer.But if we want to get better accuracy,higher the target_size=(128,128),that's why,we get lot more pixels in rows and columns in input images."""
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
"""Put the pooled feature map in a single one dimensional vector,each pooled feature map or high number is one specific feature of image like nose,ear etc"""

classifier.add(Flatten())

# Step 4 - Full connection
"""All input layer images are composed to form a single fully connected image,thus many fully connected images are formed which make a layer called fully connected layer.

Dense(output_dim=256--no of nodes in fully connected layer,so biggest ques how many?128 for bettr result,activation=relu)

Dense(output_dim=17--no of node/class in output layer(since 17 tyes of diseases),activation='relu' for return probabilities for each class and if outcomes more than two categories,Use 'softmax' activation function)"""

classifier.add(Dense(128, activation = 'relu'))
classifier.add(Dense(7, activation = 'softmax'))


# Compiling the CNN
"""Already We added all the layer in CNN.Now compile the CNN.optimizer='adam or rmsprop or adadelta' to choose decent algorithm,loss function-loss='categorical_crossentropy' for more than 2 outcomes,metrics=['accuracy'] to choose performance metric."""

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])



# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator
"""Due to an overfitting on the training set-great result on the training_set and poor result on the test_set.So before fitting,proceed to image augmentation process on images that allows us to enrich our training set for preventing overfitting.google-keras documentation.2 types-flow(X,y) and flow_from_directory(directory).train_generator and validation_generator--target size is the size of images that is expected in cnn model,we put 64 by 64.class_mode='categorical' """


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size = (64, 64),
                                                batch_size = 32,
                                                class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')


""""model.fit_generator(training_set,samples_per_epoch = no of images in training_set,  nb_epoch= 25, validation_data = validation_generator,nb_val_samples = 2000) and CNN model in not called model,its classifier"""

classifier.fit_generator(training_set,
                         steps_per_epoch = len(training_set),
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = len(test_set))


#Save model
fname = "weights.h5"
classifier.save_weights(fname, overwrite = True)

classifier.save('model.h5')

"""Here,I create a new model named classifier2.Then I use classifier2.load_weights() for new testing purpose.
classifier2 = Sequential([
    (Dense(128, activation = 'relu')),
    (Dense(4, activation = 'softmax'))
    ])
"""


#Load model

fname = "weights.h5"
classifier.load_weights(fname)

"""from keras.models import load_model
classifier.load_model('cp_image.h5')
classifier.load_weights('cp_image.h5')
"""

"""#Save and Load model with another function.
#load_model operation is equal to load_weights + load_json_architecture.
classifier.save("cp_image.h5")
from keras.models import load_model
new_model = load_model("cp_image.h5") """


#Display image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread('dataset/valid/BACTERIALBLIGHT_001.jpg')
imgplot = plt.imshow(img)
plt.show()

#What type of diseases will be predicted?

test_set.class_indices
test_set.class_indices.items()

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/valid/BACTERIALBLIGHT_001.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
#result = classifier.predict_on_batch(test_image)
#result = classifier.predict_classes(test_image)
#print(result)

#print(result.argmax())
for category, value in test_set.class_indices.items():
            if value == result.argmax():
                print(category)

#PotatoHealthy1.JPG
#PotatoEarlyBlight5.JPG
#Summary of the model               
classifier.summary()

