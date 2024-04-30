from keras.applications import vgg16


img_rows, img_cols = 224, 224


vgg = vgg16.VGG16(weights = 'imagenet',
                 include_top = False,
                 input_shape = (img_rows, img_cols, 3))

# Here we freeze the last 4 layers
# Layers are set to trainable as True by default
for layer in vgg.layers:
    layer.trainable = False

# Let's print our layers
for (i,layer) in enumerate(vgg.layers):
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)

from tensorflow.keras.layers import Conv2D, Dense, Reshape, Multiply

def spatial_attention(input_feature_maps):
    squeeze = Conv2D(filters=1, kernel_size=1, activation='sigmoid')(input_feature_maps)
    output_feature_maps = Multiply()([input_feature_maps, squeeze])
    return output_feature_maps
from keras.layers import Dense, Reshape, Permute, Activation, Dot
from keras.layers import multiply, Lambda
import keras.backend as K

def self_attention(input_feature_maps):
    # Extract the number of channels/filters
    channels = int(input_feature_maps.shape[-1])

    # Compute the attention score
    attention = Dense(channels)(input_feature_maps)
    attention = Reshape((-1, channels))(attention)  # Reshape to (height*width, channels)
    attention = Activation('softmax')(attention)

    # Apply the attention score to the input feature maps
    attention = Reshape((K.int_shape(input_feature_maps)[1], K.int_shape(input_feature_maps)[2], channels))(attention)
    output_feature_maps = multiply([input_feature_maps, attention])

    return output_feature_maps

def lw(bottom_model, num_classes):
    """creates the top or head of the model that will be
    placed ontop of the bottom layers"""

    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    top_model = Dense(512,activation='relu')(top_model)
    top_model = Dense(num_classes,activation='softmax')(top_model)
    return top_model

# Get output from the last convolutional layer of VGG16
conv_output = vgg.layers[-1].output

# Apply spatial attention mechanism
attention_output = spatial_attention(conv_output)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D

from keras.models import Model


num_classes = 2

FC_Head = lw(vgg, num_classes)

model = Model(inputs = vgg.input, outputs = FC_Head)

print(model.summary())
