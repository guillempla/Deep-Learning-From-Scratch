# Deep learning from scratch

Implementation of a feed-forward neural network in C++ from scratch. This is a task of the Neural Networks course of Masaryk University (Brno, Czech Republic). It reaches at least 88% of correct test predictions (overall accuracy). It uses L2 regularization. In order to be faster it's been implemented using **OpenMP** (16 threads).



## Data

The dataset is [**Fashion MNIST [0]**](https://www.kaggle.com/zalando-research/fashionmnist), a modern version of a well-known **MNIST [1]**. **Fashion-MNIST** is a dataset of *Zalando*'s article images ‒ consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. There are four data files — two data files as input vectors and two data files with a list of expected predictions.

The implementation exports vectors of test predictions. Such number on i-th line represents predicted class index (there are classes 0 - 9 for Fashion MNIST) for  i-th input vector. Exported files are called ***actualTestPredictions***.



## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.



### Clone the project

You can clone the project using:

```bash
git clone https://github.com/guillempla/Fashion-MNIST-Neural-Network
```



### Running

You can compile, execute and export everything with:

```bash
./RUN
```

It will read data, train the model, and generate the predictions file called ***actualPredictions*.**

> Note: You can remove "module add gcc-10.2" from "RUN". It's only necessary for tesing on AISA computer



### Sources

- [0] https://arxiv.org/pdf/1708.07747.pdf
- [1] http://yann.lecun.com/exdb/mnist/



## Author

* **Guillem Pla Bertran** - [guillempla@protonmail.com]()
