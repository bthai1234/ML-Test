import tensorflow as tf 
from tensorflow.keras import layers
from tensorflow.keras.models import Model

def build(num_classes=1):
    baseModel = tf.keras.applications.resnet_v2.ResNet50V2(
        include_top=False,
        weights="imagenet",
        input_shape=(64,64,3),
        pooling="avg",
    )

    model = baseModel.output

    model = layers.Flatten()(model)
    output = layers.Dense(num_classes, activation='softmax')(model)

    finalModel = Model(inputs=baseModel.input, outputs=output)
    return finalModel




