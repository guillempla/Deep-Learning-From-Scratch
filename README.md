# PV021 project | deep learning from scratch
**This file may not be up-to-date; read forum, emails, etc.**

### Consider using this folder structure (data, src, etc.)

Your task is simple. Implement a feed-forward neural network in C/C++ (or Rust/Swift with limited language support) and train it on a given dataset using a backpropagation algorithm. The dataset is Fashion MNIST [0],  a modern version of a well-known MNIST [1]. Fashion-MNIST is a dataset of Zalando's article images ‒ consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. The dataset in CSV format will be uploaded in IS in study materials. There are four data files — two data files as input vectors and two data files with a list of expected predictions.
The deadline is set for January 6th, 2021.
Some rules about implementations:

1. Your implementation must be compilable and runnable on the AISA server.
    1. If you know what you are doing and dare to implement the project for a GPU, please email me first [2].
2. The project must contain a runnable script called "RUN" which compiles,
executes and exports everything in "one-click".
3. The implementation must export vectors of train and test predictions.
    3. Exported predictions must be in the same format as is
"actualPredictionsExample" file ‒ on each line is only one number present.
Such number on i-th line represents predicted class index (there are classes
0 - 9 for Fashion MNIST) for i-th input vector; hence prediction order is relevant.
    3. Name the exported files "trainPredictions" and "actualTestPredictions".
4. The implementation will take both train and test input vectors, but it must
not touch test data except the evaluation of the already trained model.
    4. If any implementation will touch given test data before the evaluation
of the already trained model, it will be automatically marked as a failed
project.
    4. Why is that ‒ an optimal scenario would hide for you any test data, but
in that case, you would have to deal with serialization and deserialization of
your implementations, or you would have to be bound to some given interface and
this is just not wanted in this course.
    4. Don't cheat. Your implementations will be checked by hand.
    4. Please write doc-strings where reasonable (high-level functions, complicated functions with unusual names). The documentation will not be judged. But it's a good practice, you do not work alone, and it'll make reading your implementation easier.
5. It's demanded to reach at least 88% of correct test predictions
(overall accuracy) with given at most half an hour of training time on the Aisa machine.
    5. Implementations will be executed for a little longer, let's say for 35
minutes. At that time, it should be able to load the data, process them,
train the model, and export train/test predictions out to files.
6. The correctness will be checked using an independent program, which will be
also provided for your own purposes.
7. The use of high-level libraries is forbidden. In fact, you don't need any.
    7. Of course, you can use low-level libraries. You definitely can use basic
math functions like exp, sqrt, log, rand, etc. High-level libraries are libraries containing matrix-based operations, neural network tools such as already implemented layers with activation functions, automatic differentiation, equation/linear-program solvers, etc.
8. What you do internally with the training dataset is up to you.
9. Pack all data with your implementations and put them on the right path so
your program will load them correctly on AISA (project dir is fine).
You can make your own implementation or you can make teams of two. If there are
any problems, don't hesitate to contact me [2]. If you are struggling with
network performance, contact me.
10. Don't post your code openly on git[hub|lab] and don't read solutions already there. It's against the spirit of this project, and we compare the code already published.

---

Please, do note that my time is also limited, and this project is not the easiest
one, so start "as asap as possible, and if you encounter any problems, contact me
immediately. I won't have time for half of you day before the deadline.

---

## Some tips:
- solve the XOR problem first. XOR is a very nice example as a benchmark of the
working learning process with at least one hidden layer. Btw, the presented network
solving XOR in the lecture is minimal and it can be hard to find, so consider
more neurons in the hidden layer. If you can't solve the XOR problem, you can't
solve Fashion MNIST.
- reuse memory. You are implementing an iterative process, so don't allocate new
vectors and matrices all the time. An immutable approach is nice but very
inappropriate. Also ‒ don't copy data in some cache all the time; use indexes.
- objects are fine, but be careful about the depth of object hierarchy you are
going to create. Always remember that you are trying to be fast.
- double precision is fine. You may try to use floats. Do not use BigDecimals or
any other high precision objects.
- simple SGD is not strong, and fast enough, you are going to need to implement some
heuristics as well (or maybe not, but it's highly recommended). I suggest
heuristics: momentum, weight decay, dropout. If you are brave enough, you can
try RMSProp/AdaGrad/Adam.
- start with smaller networks and increase network topology carefully.
- consider validation of the model using part of the train dataset.
- play with hyperparameters to increase your internal validation accuracy.

## Even more tips from previous projects:
- do not wait till the week before the deadline
- .exe files are not runnable on Aisa ... :)
    - Aisa runs on "Red Hat Enterprise Linux Server release 7.5 (Maipo)"
- Aisa has 4× 16 cores, OpenMP or similar easy parallelism may help (in case you use it, please leave some cores to other applications ~ use less than 49 cores)
- missing or non-functional RUN script means that evaluation of your
implementation is not possible. So try to execute your RUN script on AISA
before your submission.
- if you are having a problem with missing compilers/maven on Aisa, you can
add such tools by including modules [3]. Please, do note, that if your
implementation requires such modules, your RUN script must include them as well,
otherwise, the RUN script won't work, and I will have no clue what to include.
- do not shuffle testing data. It won't fit expected predictions.

## FAQ:
 - "Can I write in Python, please, please, pretty please?"
	- no + it's too slow without matrix libs.
 - "Can I instead of the feed-forward implement a convolutional
 neural network?"
	- yes, but it might be much harder.
 - "Can I instead of the feed-forward implement attention?"
	- yes, I would love to see such a solution but it might be very, very hard.
 - "Have Java implementations chance against C implementations?"
	- yes. At least one of the best performing implementations was written in java.

Best luck with the project,

```python
Ronald Luc
235313@mail.muni.cz
PV021 Neural networks
```
- [0] https://arxiv.org/pdf/1708.07747.pdf
- [1] http://yann.lecun.com/exdb/mnist/
- [2] Ronald Luc, 235313
- [3] https://www.fi.muni.cz/tech/unix/modules.html.en
