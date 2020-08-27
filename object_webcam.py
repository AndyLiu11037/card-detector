import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import sys
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import warnings
import cv2
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings
matplotlib.use('TKAgg', force=True)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

PATH_TO_SAVED_MODEL = os.path.join(os.getcwd(),"my_model_2")
print('Loading model...', end='')

start_time = time.time()
category_index = label_map_util.create_category_index_from_labelmap(os.path.join(os.getcwd(),"labelmap.pbtxt"),
                                                                    use_display_name=True)
# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_SAVED_MODEL+"/pipeline.config")
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)
# Load saved model and build the detection function
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_SAVED_MODEL+"/checkpoint", 'ckpt-0')).expect_partial()
    
end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))

print('Running inference for video source... ', end='')

video = cv2.VideoCapture(0) #depending on which webcame/videosource you want 0 is default

while True:
  ret, image_np = video.read()
  image_np_expanded = np.expand_dims(image_np, axis=0)

  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  detections, predictions_dict, shapes = detect_fn(input_tensor)

  # input_tensor = np.expand_dims(image_np, 0)

  label_id_offset = 1
  image_np_with_detections = image_np.copy()
  viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'][0].numpy(),
        (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False)
  cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))
  if cv2.waitKey(25) & 0xFF == ord('q'):
      break
video.release()
cv2.destroyAllWindows()
