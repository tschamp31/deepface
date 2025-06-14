# built-in dependencies
import os
import subprocess
from functools import partial
from typing import Optional
import zipfile
import bz2

# 3rd party dependencies
import gdown
import keras
import cupy as np
import onnx
import tf2onnx

from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.keras import Model
from tensorflow.python.keras.models import save_model

# project dependencies
from deepface.commons import folder_utils, package_utils
from deepface.commons.logger import Logger

tf_version = package_utils.get_tf_major_version()
if tf_version == 1:
    from keras.models import Sequential
else:
    from tensorflow.keras import Sequential
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

logger = Logger()

# pylint: disable=line-too-long, use-maxsplit-arg

ALLOWED_COMPRESS_TYPES = ["zip", "bz2"]

def download_weights_if_necessary(
    file_name: str, source_url: str, compress_type: Optional[str] = None
) -> str:
    """
    Download the weights of a pre-trained model from external source if not downloaded yet.
    Args:
        file_name (str): target file name with extension
        source_url (url): source url to be downloaded
        compress_type (optional str): compress type e.g. zip or bz2
    Returns
        target_file (str): exact path for the target file
    """
    home = folder_utils.get_deepface_home()

    target_file = os.path.normpath(os.path.join(home, ".deepface/weights", file_name))

    if os.path.isfile(target_file):
        logger.debug(f"{file_name} is already available at {target_file}")
        return target_file

    if compress_type is not None and compress_type not in ALLOWED_COMPRESS_TYPES:
        raise ValueError(f"unimplemented compress type - {compress_type}")

    try:
        logger.info(f"🔗 {file_name} will be downloaded from {source_url} to {target_file}...")

        if compress_type is None:
            gdown.download(source_url, target_file, quiet=False)
        elif compress_type is not None and compress_type in ALLOWED_COMPRESS_TYPES:
            gdown.download(source_url, f"{target_file}.{compress_type}", quiet=False)

    except Exception as err:
        raise ValueError(
            f"⛓️‍💥 An exception occurred while downloading {file_name} from {source_url}. "
            f"Consider downloading it manually to {target_file}."
        ) from err

    # uncompress downloaded file
    if compress_type == "zip":
        with zipfile.ZipFile(f"{target_file}.zip", "r") as zip_ref:
            zip_ref.extractall(os.path.join(home, ".deepface/weights"))
            logger.info(f"{target_file}.zip unzipped")
    elif compress_type == "bz2":
        bz2file = bz2.BZ2File(f"{target_file}.bz2")
        data = bz2file.read()
        with open(target_file, "wb") as f:
            f.write(data)
        logger.info(f"{target_file}.bz2 unzipped")

    return target_file


def load_model_weights(model: Model, weight_file: str) -> Model:
    """
    Load pre-trained weights for a given model
    Args:
        model (keras.models.Model): pre-built model
        weight_file (str): exact path of pre-trained weights
    Returns:
        model (keras.models.Sequential): pre-built model with
            updated weights
    """
    try:
        model.load_weights(weight_file)
    except Exception as err:
        raise ValueError(
            f"An exception occurred while loading the pre-trained weights from {weight_file}."
            "This might have happened due to an interruption during the download."
            "You may want to delete it and allow DeepFace to download it again during the next run."
            "If the issue persists, consider downloading the file directly from the source "
            "and copying it to the target folder."
        ) from err
    return model

def convert_model_to_saved_model(model: Model, model_name: str, build_batch_size=10):
    home = folder_utils.get_deepface_home()
    pb_path = os.path.normpath(os.path.join(home, ".deepface/weights", model_name, "tf_saved_model"))
    os.makedirs(os.path.join(pb_path), exist_ok=True)
    pb_path_file = os.path.join(pb_path, "saved_model.pb")
    if os.path.exists(pb_path_file):
        return
    else:
        print("pb path file does not exist")
        tf.saved_model.save(model, pb_path, signatures=tf.function(model, input_signature=[tf.TensorSpec(model.input_shape, name="input")]).get_concrete_function())

def convert_model_to_onnx(model: Model, model_name: str, build_batch_size=10):
    home = folder_utils.get_deepface_home()
    onnx_path = os.path.normpath(os.path.join(home, ".deepface/weights", model_name, "onnx"))
    saved_model = os.path.normpath(os.path.join(home, ".deepface/weights", model_name, "saved_model"))
    os.makedirs(os.path.join(onnx_path), exist_ok=True)
    os.makedirs(os.path.join(saved_model), exist_ok=True)
    onnx_path_file = os.path.join(onnx_path, "saved_model.pb")
    if os.path.exists(onnx_path_file):
        return
    else:
        print("onnx path file does not exist")
        print(f"InputShape: {model.input_shape}")
        tf.saved_model.save(model, signatures=tf.function(model, input_signature=[tf.TensorSpec(model.input_shape, name="input")]).get_concrete_function(),export_dir=saved_model)

        # use tf2onnx to convert to onnx model
        cmd = 'python -m tf2onnx.convert --saved-model {} --output {} --opset {}'.format(saved_model,onnx_path_file, 18)
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()


def convert_model_to_trt_pb(model: Model, model_name: str, build_batch_size: int = 10) -> None:
    home = folder_utils.get_deepface_home()
    trt_path = os.path.normpath(os.path.join(home, ".deepface/weights", model_name, "trt"))
    onnx_path = os.path.normpath(os.path.join(home, ".deepface/weights", model_name, "saved_model"))
    os.makedirs(os.path.join(trt_path), exist_ok=True)
    if os.path.exists(os.path.join(trt_path, "saved_model.pb")):
        return
    else:
        def input_fnc(batch_size, inputs):
            image_shape = (inputs[1], inputs[2], 3)
            # image_shape = (3, imgsz, imgsz)
            data_shape = (batch_size,) + image_shape

            for _ in range(100):
                img = np.random.uniform(-1, 1, size=data_shape).astype("float32").get()
                yield (img,)

        converter = trt.TrtGraphConverterV2(input_saved_model_dir=onnx_path, precision_mode=trt.TrtPrecisionMode.FP32,minimum_segment_size=3,max_workspace_size_bytes=6000000000)
        converter.convert()
        converter.build(input_fn=partial(input_fnc, build_batch_size, model.input_shape))
        converter.save(output_saved_model_dir=trt_path, save_gpu_specific_engines=True)


def download_all_models_in_one_shot() -> None:
    """
    Download all model weights in one shot
    """

    # import model weights from module here to avoid circular import issue
    from deepface.models.facial_recognition.VGGFace import WEIGHTS_URL as VGGFACE_WEIGHTS
    from deepface.models.facial_recognition.Facenet import FACENET128_WEIGHTS, FACENET512_WEIGHTS
    from deepface.models.facial_recognition.OpenFace import WEIGHTS_URL as OPENFACE_WEIGHTS
    from deepface.models.facial_recognition.FbDeepFace import WEIGHTS_URL as FBDEEPFACE_WEIGHTS
    from deepface.models.facial_recognition.ArcFace import WEIGHTS_URL as ARCFACE_WEIGHTS
    from deepface.models.facial_recognition.DeepID import WEIGHTS_URL as DEEPID_WEIGHTS
    from deepface.models.facial_recognition.SFace import WEIGHTS_URL as SFACE_WEIGHTS
    from deepface.models.facial_recognition.GhostFaceNet import WEIGHTS_URL as GHOSTFACENET_WEIGHTS
    from deepface.models.facial_recognition.Dlib import WEIGHT_URL as DLIB_FR_WEIGHTS
    from deepface.models.demography.Age import WEIGHTS_URL as AGE_WEIGHTS
    from deepface.models.demography.Gender import WEIGHTS_URL as GENDER_WEIGHTS
    from deepface.models.demography.Race import WEIGHTS_URL as RACE_WEIGHTS
    from deepface.models.demography.Emotion import WEIGHTS_URL as EMOTION_WEIGHTS
    from deepface.models.spoofing.FasNet import (
        FIRST_WEIGHTS_URL as FASNET_1ST_WEIGHTS,
        SECOND_WEIGHTS_URL as FASNET_2ND_WEIGHTS,
    )
    from deepface.models.face_detection.Ssd import (
        MODEL_URL as SSD_MODEL,
        WEIGHTS_URL as SSD_WEIGHTS,
    )
    from deepface.models.face_detection.Yolo import (
        WEIGHT_URLS as YOLO_WEIGHTS,
        WEIGHT_NAMES as YOLO_WEIGHT_NAMES,
        YoloModel
    )
    from deepface.models.face_detection.YuNet import WEIGHTS_URL as YUNET_WEIGHTS
    from deepface.models.face_detection.Dlib import WEIGHTS_URL as DLIB_FD_WEIGHTS
    from deepface.models.face_detection.CenterFace import WEIGHTS_URL as CENTERFACE_WEIGHTS

    WEIGHTS = [
        # facial recognition
        VGGFACE_WEIGHTS,
        FACENET128_WEIGHTS,
        FACENET512_WEIGHTS,
        OPENFACE_WEIGHTS,
        FBDEEPFACE_WEIGHTS,
        ARCFACE_WEIGHTS,
        DEEPID_WEIGHTS,
        SFACE_WEIGHTS,
        {
            "filename": "ghostfacenet_v1.h5",
            "url": GHOSTFACENET_WEIGHTS,
        },
        DLIB_FR_WEIGHTS,
        # demography
        AGE_WEIGHTS,
        GENDER_WEIGHTS,
        RACE_WEIGHTS,
        EMOTION_WEIGHTS,
        # spoofing
        FASNET_1ST_WEIGHTS,
        FASNET_2ND_WEIGHTS,
        # face detection
        SSD_MODEL,
        SSD_WEIGHTS,
        {
            "filename": YOLO_WEIGHT_NAMES[YoloModel.V8N.value],
            "url": YOLO_WEIGHTS[YoloModel.V8N.value],
        },
        {
            "filename": YOLO_WEIGHT_NAMES[YoloModel.V11N.value],
            "url": YOLO_WEIGHTS[YoloModel.V11N.value],
        },
        {
            "filename": YOLO_WEIGHT_NAMES[YoloModel.V11S.value],
            "url": YOLO_WEIGHTS[YoloModel.V11S.value],
        },
        {
            "filename": YOLO_WEIGHT_NAMES[YoloModel.V11M.value],
            "url": YOLO_WEIGHTS[YoloModel.V11M.value],
        },
        YUNET_WEIGHTS,
        DLIB_FD_WEIGHTS,
        CENTERFACE_WEIGHTS,
    ]

    for i in WEIGHTS:
        if isinstance(i, str):
            url = i
            filename = i.split("/")[-1]
            compress_type = None
            # if compressed file will be downloaded, get rid of its extension
            if filename.endswith(tuple(ALLOWED_COMPRESS_TYPES)):
                for ext in ALLOWED_COMPRESS_TYPES:
                    compress_type = ext
                    if filename.endswith(f".{ext}"):
                        filename = filename[: -(len(ext) + 1)]
                        break
        elif isinstance(i, dict):
            filename = i["filename"]
            url = i["url"]
        else:
            raise ValueError("unimplemented scenario")
        logger.info(
            f"Downloading {url} to ~/.deepface/weights/{filename} with {compress_type} compression"
        )
        download_weights_if_necessary(
            file_name=filename, source_url=url, compress_type=compress_type
        )