# Pygame Object Detection

## Introduction

Object detection based on [KleinYuan/tf-object-detection](https://github.com/KleinYuan/tf-object-detection), implemented using pre-trained models on tensorflow and pygame.


## Dependencies:

* Tensorflow >= 1.4.1
* Pygame >= 1.9.3


## Running

1. Download and extract a model (e.g. ssd_mobilenet_v1_coco_xxxx_xx_xx)
[Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

2. Setting up dependencies
```
pip install --upgrade tensorflow pygame
```

3. Edit model information in *pod.py* (configured for ssd_mobilenet_v1_coco_2017_11_17)

4. Run
```
python pod.py
```


## Credits

* @KleinYuan