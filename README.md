# DenseNet-Tensorflow
An refactor of densenet in low level tensorflow.

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

2. Train example data:
```bash
cd densenet-tensorflow
```
```python
python train.py
```

3. Visualize training loss:
```bash
tensorboard --logdir=./log
```

4. Test model:
```python
python test.py
```
## Reference
[Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)



Still under construction.
