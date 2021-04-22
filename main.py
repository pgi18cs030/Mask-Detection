from keras.models import  Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier=Sequential()

classifier.add(Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Conv2D(32,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Conv2D(32,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(units=158,activation='relu'))
classifier.add(Dense(units=2,activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1./255)

test_datagen=ImageDataGenerator(rescale=1./255)

training_set=train_datagen.flow_from_directory('C:/Users/hp/Desktop/new mask/training',
                                               target_size=(128,128),
                                               batch_size=32,
                                               )

test_set=test_datagen.flow_from_directory('C:/Users/hp/Desktop/new mask/test',
                                               target_size=(128,128),
                                               batch_size=32,
                                               )

test1=training_set.class_indices
test1

classifier.fit_generator(training_set,steps_per_epoch=100,epochs=70,validation_data=test_set,validation_steps=200)


classifier.save('C:/Users/hp/Desktop/new mask/final.h5')

from keras.models import load_model
new_model=load_model('C:/Users/hp/Desktop/new mask/final.h5')

new_model.summary()
#from keras.preprocessing import image
#test=image.load_img('C:/Users/hp/Desktop/new mask/mask1.jpg',target_size=(128,128))
#test=image.img_to_array(test)
#import numpy as np
#test=np.expand_dims(test,axis=0)
#test/=255
#test
#prediction=new_model.predict(test)
#max_index=np.argmax(prediction)
#max_index
#face_mask=['with_mask','without_mask']
#face_mask=face_mask[max_index]