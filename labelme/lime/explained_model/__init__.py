import collections
import keras

Model = collections.namedtuple(
    "Model", ["name", "shape"]
)

MODELS = [
    Model(
        name="InceptionV3", shape=299,
    ),
    Model(
        name="InceptionResNetV2", shape=299,
    ),
    Model(
        name="Xception",  shape=299,
    ),

    Model(
        name="ResNet50",  shape=224,
    ),
    Model(
        name="ResNet101",  shape=224,
    ),
    Model(
        name="ResNet152V2", shape=224,
    ),

    Model(
        name="MobileNet",  shape=224,
    ),
    Model(
        name="MobileNetV2", shape=224,
    ),
]

