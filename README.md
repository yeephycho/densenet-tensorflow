# DenseNet-Tensorflow
A refactor of densenet in low level tensorflow API.

No tf.slim

No tf.layers

No tf.contrib

No opencv

Current graph is based on DenseNet-BC-121 with 224 input image size.

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
## Reference
[Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

