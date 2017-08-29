# DenseNet-Tensorflow
A refactor of densenet in low level tensorflow API.

No tf.slim

No tf.layers

No tf.contrib

No opencv

Current graph is a variation of DenseNet-BC-121 with 224 input image size, there's only 52 convolutional layers.

Can train and test from scratch, other features are still under construction.

## Feature
Support tfrecord

With minimum dependencies


## Dependencies
numpy

## Usage
1. Clone the repo:
```bash
git clone https://github.com/yeephycho/densenet-tensorflow.git
```

2. Download example tfrecord data:
Click [here](https://drive.google.com/drive/folders/0BwTYOWiLy2btX2RiZHlDYVdiWVE?usp=sharing) to download.
About how to generate tfrecord, please see [repo.](https://github.com/yeephycho/tensorflow_input_image_by_tfrecord) or see the script from tensorflow [build image data](https://github.com/tensorflow/models/blob/master/inception/inception/data/build_image_data.py).

Data comes from tensorflow [inception retraining example](https://github.com/tensorflow/models/tree/master/inception) which contains 5 kinds of flowers, click [here](http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz) to download original data.

3. Train example data:
```bash
cd densenet-tensorflow
```
```python
python train.py
```

4. Visualize training loss:
```bash
tensorboard --logdir=./log
```

5. Test model:
```python
python test.py
```

Expected accuracy should be around 80%.
![Result](https://github.com/yeephycho/densenet-tensorflow/blob/master/res/test_result.png?raw=true "Show result")


## Reference
[Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

