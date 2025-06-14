# built-in dependencies
from typing import List
import os

# 3rd party dependencies
import cupy as np
import tensorflow.keras

# project dependencies
from deepface.commons import package_utils, weight_utils, folder_utils
from deepface.modules import verification
from deepface.models.FacialRecognition import FacialRecognition
from deepface.commons.logger import Logger

logger = Logger()

# ---------------------------------------

tf_version = package_utils.get_tf_major_version()
if tf_version == 1:
    from keras.models import Model, Sequential
    from keras.layers import (
        Convolution2D,
        ZeroPadding2D,
        MaxPooling2D,
        Flatten,
        Dropout,
        Activation,
    )
    USE_PB = False
else:
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import (
        Convolution2D,
        ZeroPadding2D,
        MaxPooling2D,
        Flatten,
        Dropout,
        Activation,
    )
    from tensorflow.compiler.tf2tensorrt._pywrap_py_utils import is_tensorrt_enabled
    if is_tensorrt_enabled():
        import tensorflow.python.compiler.tensorrt.trt_convert as trt
        import tensorrt as trt_runtime
        print(trt_runtime)
        print(dir(trt_runtime))
        if trt is not None:
            USE_PB = True
    else:
        USE_PB = False
# ---------------------------------------

WEIGHTS_URL = (
    "https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5"
)


# pylint: disable=too-few-public-methods
class VggFaceClient(FacialRecognition):
    """
    VGG-Face model class
    """

    def __init__(self):
        self.model_name = "VGG-Face"
        self.input_shape = (224, 224)
        self.output_shape = 4096
        self.model = self.load_model()

    def forward(self, img: np.ndarray) -> List[float]:
        """
        Generates embeddings using the VGG-Face model.
            This method incorporates an additional normalization layer.

        Args:
            img (np.ndarray): pre-loaded image in BGR
        Returns
            embeddings (list): multi-dimensional vector
        """
        # model.predict causes memory issue when it is called in a for loop
        # embedding = self.model.predict(img, verbose=0)

        # having normalization layer in descriptor troubles for some gpu users (e.g. issue 957, 966)
        # instead we are now calculating it with traditional way not with keras backend
        embedding = self.model(img, training=False).numpy()
        if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
            embedding = verification.l2_normalize(embedding, axis=1)
        else:
            embedding = verification.l2_normalize(embedding)
        return embedding.tolist()


    def load_model(self) -> Model:
        """
        Base model of VGG-Face being used for classification - not to find embeddings
        Returns:
            model (Sequential): model was trained to classify 2622 identities
        """
        layers = Sequential()
        layers.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
        layers.add(Convolution2D(64, (3, 3), activation="relu"))
        layers.add(ZeroPadding2D((1, 1)))
        layers.add(Convolution2D(64, (3, 3), activation="relu"))
        layers.add(MaxPooling2D((2, 2), strides=(2, 2)))

        layers.add(ZeroPadding2D((1, 1)))
        layers.add(Convolution2D(128, (3, 3), activation="relu"))
        layers.add(ZeroPadding2D((1, 1)))
        layers.add(Convolution2D(128, (3, 3), activation="relu"))
        layers.add(MaxPooling2D((2, 2), strides=(2, 2)))

        layers.add(ZeroPadding2D((1, 1)))
        layers.add(Convolution2D(256, (3, 3), activation="relu"))
        layers.add(ZeroPadding2D((1, 1)))
        layers.add(Convolution2D(256, (3, 3), activation="relu"))
        layers.add(ZeroPadding2D((1, 1)))
        layers.add(Convolution2D(256, (3, 3), activation="relu"))
        layers.add(MaxPooling2D((2, 2), strides=(2, 2)))

        layers.add(ZeroPadding2D((1, 1)))
        layers.add(Convolution2D(512, (3, 3), activation="relu"))
        layers.add(ZeroPadding2D((1, 1)))
        layers.add(Convolution2D(512, (3, 3), activation="relu"))
        layers.add(ZeroPadding2D((1, 1)))
        layers.add(Convolution2D(512, (3, 3), activation="relu"))
        layers.add(MaxPooling2D((2, 2), strides=(2, 2)))

        layers.add(ZeroPadding2D((1, 1)))
        layers.add(Convolution2D(512, (3, 3), activation="relu"))
        layers.add(ZeroPadding2D((1, 1)))
        layers.add(Convolution2D(512, (3, 3), activation="relu"))
        layers.add(ZeroPadding2D((1, 1)))
        layers.add(Convolution2D(512, (3, 3), activation="relu"))
        layers.add(MaxPooling2D((2, 2), strides=(2, 2)))

        layers.add(Convolution2D(4096, (7, 7), activation="relu"))
        layers.add(Dropout(0.5))
        layers.add(Convolution2D(4096, (1, 1), activation="relu"))
        layers.add(Dropout(0.5))
        layers.add(Convolution2D(2622, (1, 1)))
        layers.add(Flatten())
        layers.add(Activation("softmax"))

        weight_file = weight_utils.download_weights_if_necessary(
            file_name="vgg_face_weights.h5", source_url=WEIGHTS_URL
        )

        model = weight_utils.load_model_weights(model=layers, weight_file=weight_file)

        # 2622d dimensional model
        # vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

        # 4096 dimensional model offers 6% to 14% increasement on accuracy!
        # - softmax causes underfitting
        # - added normalization layer to avoid underfitting with euclidean
        # as described here: https://github.com/serengil/deepface/issues/944
        output_layer = Flatten()(model.layers[-5].output)
        # keras backend's l2 normalization layer troubles some gpu users (e.g. issue 957, 966)
        # base_model_output = Lambda(lambda x: K.l2_normalize(x, axis=1), name="norm_layer")(
        #     base_model_output
        # )
        #vgg_face_descriptor = Model(inputs=model.input, outputs=base_model_output)
        USE_PB = True
        if USE_PB:
            model = Model(inputs=model.inputs, outputs=output_layer)
            weight_utils.convert_model_to_onnx(model, self.model_name)
            weight_utils.convert_model_to_trt_pb(model, self.model_name)
            return model
        else:
            return Model(inputs=model.input, outputs=model.output_shape)

if __name__ == "__main__":
    USE_PB = True
    folder_utils.initialize_folder()
    test = VggFaceClient()
    print(is_tensorrt_enabled())
    for layer in test.model.layers:
        print(layer, layer.name)
