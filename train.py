from keras.models import Sequential # creating DLL structure model
from keras.layers import Conv2D# using 2 dimensional convolution filter
from keras.layers import MaxPooling2D# using 2-D max pooling filters
from keras.layers import Flatten# used for flattening the result array/image
from keras.layers import Dense# used for creating/declaring neurons
from keras.preprocessing.image import ImageDataGenerator# importing function to acces images in data set for training model


#Defining a modle
model=Sequential()
model.add(Conv2D(32,(3,3),input_shape = (64,64,3),activation = 'relu'))# defining 2-D convolution filter
model.add(MaxPooling2D(pool_size=(2,2)))# defining 2-D maxpooling filter
model.add(Flatten())# flattening result
model.add(Dense(units=128,activation='relu'))# structuring input model of ANN
model.add(Dense(units=1,activation='sigmoid'))#structuring ouytput model of ANN
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])# optimising model
# preprocessing image in dataset
train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
val_datagen=ImageDataGenerator(rescale=1./255)# testing dataset
#accesing the images in the dataset via sub folders train,val
training_set=train_datagen.flow_from_directory('Dataset/train',target_size=(64,64),batch_size=10,class_mode='binary')
val_set=val_datagen.flow_from_directory('Dataset/val',target_size=(64,64),batch_size=10,class_mode='binary')

model.fit_generator(training_set,steps_per_epoch=10,epochs=25,validation_data=val_set,validation_steps=2)
# saving the model
model_json=model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("saved model to disk")

