# DenseNet-Tensorflow
An refactor of densenet in low level tensorflow.

Current model is DenseNet-BC-121 with 224 input image size.

Still under construction.

## Feature
No tf.slim

No tf.layers

No tf.contrib

No opencv

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
## Reference
[Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

