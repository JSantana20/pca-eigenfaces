# The following model is being developed based on the article published by Ronneberger, Fischer and Brox
# U-Net: Convolutional Networks for Biomedical Image Segmentation

import tensorflow as tf
import tensorflow.keras.optimizers as opts
import tensorflow.keras.regularizers as regs
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,InputLayer,Conv2D,Conv2DTranspose,\
                                    MaxPooling2D,BatchNormalization,Dropout,\
                                    Dense,CenterCrop,UpSampling2D,Layer

# Contracting model
def cmodel(input_layer,num_filters):
  cm1 = Conv2D(num_filters,(3,3),activation='relu',padding='valid')(input_layer)
  cm2 = Conv2D(num_filters,(3,3),activation='relu',padding='valid')(cm1)
  crop = CenterCrop(crop_size[0],crop_size[1])(cm2)
  out = MaxPooling2D((2,2))(cm2)
  return out,crop

# Expanding model
def emodel(input_layer,num_filters):
  em1 = Conv2DTranspose(num_filters,(2,2),(2,2),padding='valid')(input_layer)
  em2 = tf.concat([concat_layer,em1],3)  #otherwise, 2
  em3 = Conv2D(num_filters,(3,3),activation='relu',padding='valid')(em2)
  out = Conv2D(num_filters,(3,3),activation='relu',padding='valid')(em3)
  return out

## U-Net
# Contracting path
in_layer = Input((572,572,3))
x,t0 = cmodel(in_layer,64,(392,392))
x,t1 = cmodel(x,128,(200,200))
x,t2 = cmodel(x,256,(104,104))
x,t3 = cmodel(x,512,(56,56))
x = Conv2D(1024,(3,3),activation='relu',padding='valid')(x)
x = Conv2D(1024,(3,3),activation='relu',padding='valid')(x)

# Expanding path
x = emodel(x,t3,512)
x = emodel(x,t2,256)
x = emodel(x,t1,128)
x = emodel(x,t0,64)
out_layer = Conv2D(2,(1,1),padding='valid')(x)

unet = Model(inputs=in_layer,outputs=out_layer)

unet.summary()

# Model compilation
unet.compile(optimizer=opts.Adam(learning_rate=0.0001),loss='binary_crossentropy',metrics=['accuracy','mae'])

# Callbacks definition
class TrainingCallback(Callback):
  def on_train_begin(self,logs=None):
    print('Starting training...')
  def on_epoch_begin(self,epoch,logs=None):
    print(f"Starting epoch {epoch+1}")
  def on_epoch_end(self,epoch,logs=None):
    print(f"Finished epoch {epoch+1}")
  def on_train_end(self,logs=None):
    print('Finished training')

checkpoint_path = '/content/gdrive/MyDrive/Semillero/checkpoint'
checkpoint = ModelCheckpoint(checkpoint_path,frequency='epoch',
save_weights_only=1,save_best_only=1,verbose=1,monitor='val_accuracy')

early_stopping = EarlyStopping(monitor='val_accuracy',min_delta=1e-3,patience=10)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',factor=0.5,patience=2,\
                              verbose=1,min_delta=1e-3)

# Model training
history = unet.fit(train_data,train_labels,epochs=50,batch_size=16,
                    validation_data=(validation_data,validation_labels),verbose=2),
                    callbacks=[checkpoint,TrainingCallback(),early_stopping,reduce_lr])
