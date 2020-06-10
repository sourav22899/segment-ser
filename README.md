# segment-ser
This is the offical code repo for [A Segment Level Approach to Speech Emotion Recognition using Transfer Learning](https://link.springer.com/chapter/10.1007%2F978-3-030-41299-9_34), ACPR 2019. In this paper, we propose a speech emotion recognition system that predicts emotions for multiple segments of a single audio clip unlike the conventional emotion recognition models that predict the emotion of an entire audio clip directly. The proposed system consists of a pre-trained deep convolutional neural network (CNN), the Google [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset/vggish) model, followed by a single layered neural network which predicts the emotion classes of the audio segments. The proposed model attains an accuracy of 68.7% surpassing the current state-of-the-art models in classifying the data into one of the four emotional classes (angry, happy, sad and neutral) when trained and evaluated on IEMOCAP audio-only dataset.


## Requirements
* [`numpy`](http://www.numpy.org/)
* [`resampy`](http://resampy.readthedocs.io/en/latest/)
* [`tensorflow`](http://www.tensorflow.org/) (currently, only TF v1.x)
* [`tf_slim`](https://github.com/google-research/tf-slim)
* [`six`](https://pythonhosted.org/six/)
* [`soundfile`](https://pysoundfile.readthedocs.io/)

These are all easily installable via, e.g., `pip install numpy` 

To implement the code, the following two data files needs to be downloaded:

* [VGGish model checkpoint](https://storage.googleapis.com/audioset/vggish_model.ckpt),
  in TensorFlow checkpoint format.
* [Embedding PCA parameters](https://storage.googleapis.com/audioset/vggish_pca_params.npz),
  in NumPy compressed archive format.
After downloading these files into the same directory as this README.

## Citation
If you find this repo useful in your research, please consider to cite the following paper:

```
@inproceedings{sahoo2019segment,
  title={A Segment Level Approach to Speech Emotion Recognition Using Transfer Learning},
  author={Sahoo, Sourav and Kumar, Puneet and Raman, Balasubramanian and Roy, Partha Pratim},
  booktitle={Asian Conference on Pattern Recognition},
  pages={435--448},
  year={2019},
  organization={Springer}
}
```