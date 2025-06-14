# project dependencies
from deepface.commons import package_utils, weight_utils
from deepface.models.FacialRecognition import FacialRecognition
from deepface.commons.logger import Logger

logger = Logger()

# --------------------------------
# dependency configuration

tf_major = package_utils.get_tf_major_version()
tf_minor = package_utils.get_tf_minor_version()

if tf_major == 1:
    from keras.models import Model, Sequential
    from keras.layers import (
        Convolution2D,
        MaxPooling2D,
        Flatten,
        Dense,
        Dropout,
    )
else:
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import (
        Convolution2D,
        MaxPooling2D,
        Flatten,
        Dense,
        Dropout,
    )

# pylint: disable=line-too-long, too-few-public-methods
WEIGHTS_URL="https://github.com/swghosh/DeepFace/releases/download/weights-vggface2-2d-aligned/VGGFace2_DeepFace_weights_val-0.9034.h5.zip"

class DeepFaceClient(FacialRecognition):
    """
    Fb's DeepFace model class
    """

    def __init__(self):
        # DeepFace requires tf 2.12 or less

        self.model_name = "DeepFace"
        self.input_shape = (152, 152)
        self.output_shape = 4096
        self.model = self.load_model()

    def load_model(self,
        url=WEIGHTS_URL,
    ) -> Model:
        """
        Construct DeepFace model, download its weights and load
        """
        # we have some checks for this dependency in the init of client
        # putting this in global causes library initialization
        if tf_major == 1:
            from keras.layers import LocallyConnected2D
        else:
            from deepface.commons.locally_connected2d import LocallyConnected2D

        base_model = Sequential()
        base_model.add(
            Convolution2D(32, (11, 11), activation="relu", name="C1", input_shape=(152, 152, 3))
        )
        base_model.add(MaxPooling2D(pool_size=3, strides=2, padding="same", name="M2"))
        base_model.add(Convolution2D(16, (9, 9), activation="relu", name="C3"))
        base_model.add(LocallyConnected2D(16, (9, 9), activation="relu", name="L4"))
        base_model.add(LocallyConnected2D(16, (7, 7), strides=2, activation="relu", name="L5"))
        base_model.add(LocallyConnected2D(16, (5, 5), activation="relu", name="L6"))
        base_model.add(Flatten(name="F0"))
        base_model.add(Dense(4096, activation="relu", name="F7"))
        base_model.add(Dropout(rate=0.5, name="D0"))
        base_model.add(Dense(8631, activation="softmax", name="F8"))

        # ---------------------------------

        weight_file = weight_utils.download_weights_if_necessary(
            file_name="VGGFace2_DeepFace_weights_val-0.9034.h5", source_url=url, compress_type="zip"
        )

        base_model = weight_utils.load_model_weights(model=base_model, weight_file=weight_file)

        # drop F8 and D0. F7 is the representation layer.
        deepface_model = Model(inputs=base_model.layers[0].input, outputs=base_model.layers[-3].output)
        weight_utils.convert_model_to_saved_model(model=deepface_model, model_name=self.model_name)
        weight_utils.convert_model_to_onnx(model=deepface_model,model_name=self.model_name)
        #weight_utils.convert_model_to_trt_pb(model=deepface_model, model_name=self.model_name)
        return deepface_model

if __name__ == "__main__":
    USE_PB = True
    test = DeepFaceClient()
    for layer in test.model.layers:
        print(layer, layer.name)