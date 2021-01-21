# Source code

## TODO:

- [x] Activation functions
  - [x] Sigmoid
  - [x] Relu
  - [x] Softmax

- [x] Loss functions
  - [x] Mean square
  - [x] Mean square prime
  - [x] Cross entropy 
  - [x] Cross entropy prime
  - [ ] Binary Cross entropy  $$-\frac{1}{m} \sum\limits_{i = 1}^{m} (y^{(i)}\log\left(a^{[L]  (i)}\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}\right)) \tag{1}$$
  - [ ] Binary Cross entropy prime 
  
- [x] Data processing
  - [x] Read labels
  - [x] Read vectors
  - [x] Write predictions
  
- [x] Matrix
  - [x] Scalar operators (+,- ,*,/)
  - [x] Matrices operators (+,- ,*,/)
  - [x] Activation and Loss parallel functions
  - [x] Print
  - [x] Broadcast sum
  - [x] Random initialization
    - [x] Uniform random initialization
    - [x] Glorot (Normal distribution with $mean=0$ and $variance=\frac{2}{m+n}$)
  - [x] OpenMP implementation
  
- [x] Layer
  - [x] Initialize parameters
  
- [ ] Model
  - [x] Initialize parameters
  - [x] Train
    - [x] Feed Forward
    - [x] Back Forward
    - [x] Get info
      - [x] Calculate error
      - [x] Calculate accuracy
    - [x] Update parameters
  - [x] Predict
  - [ ] Mini-Batch
    - [ ] Random shuffle data
    - [ ] Create $\frac{num\_examples}{bach\_size}$ batches
- [ ] Fine-Tune
  - [ ] Update Hyperparameters
  - [ ] Early stop
- [x] Main
  - [x] Read data
  - [x] Train model
  - [x] Predict
  - [x] Write predictions
  - [x] Change seed each time
  
- [x] RUN
  - [x] Add modules
  - [x] Makefile
