import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv2D,Conv2DTranspose,MaxPooling2D,\
                                    BatchNormalization,Dropout,Dense,CenterCrop,\
                                    UpSampling2D,Layer

# U-Net
def InputLayer(input_shape):
  return Input(input_shape)

# Contracting model
def cmodel(input_layer,num_filters):
  cm1 = Conv2D(num_filters,(3,3),activation='relu',padding='valid')(input_layer)
  cm2 = Conv2D(num_filters,(3,3),activation='relu',padding='valid')(cm1)
  out = MaxPooling2D((2,2))(cm2)
  return out

# Expanding model
def emodel(input_layer,num_filters):
  em1 = UpSampling2D((2,2))(input_layer)
  em2 = 

# Contracting path
in_layer = tf.expand_dims(InputLayer((572,572)),0)
x = cmodel(in_layer,64)
x = cmodel(x,128)
x = cmodel(x,256)
x = cmodel(x,512)
x = Conv2D(1024,(3,3),activation='relu',padding='valid')(x)
out_cpath = Conv2D(1024,(3,3),activation='relu',padding='valid')(x)

unet = Model(inputs=in_layer,outputs=out_cpath)

unet.summary()
                                    
