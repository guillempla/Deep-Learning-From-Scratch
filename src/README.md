# Source code

## TODO:

- [x] Activation functions
  - [x] Sigmoid
  - [x] Relu
  - [ ] Softmax
    - [x] Softmax
    - [ ] Softmax prime

- [ ] Loss functions
  - [x] Mean square
  - [ ] Cross entropy

- [x] Data processing
  - [x] Read labels
  - [x] Read vectors
  - [x] Write predictions
  
- [ ] Matrix
  - [x] Scalar operators (+,- ,*,/)
  - [x] Matrices operators (+,- ,*,/)
  - [x] Activation and Loss parallel functions
  - [x] Print
  - [x] Broadcast sum
  - [ ] Random initialization
    - [x] Uniform random initialization
    - [ ] Glorot (Normal distribution with $mean=0$ and $variance=\frac{2}{numInputs+numNeurons}$)
    - [ ] Other initialization models
  - [x] OpenMP implementation
  
- [x] Layer
  - [x] Initialize parameters
  - [x] Feed Forward
  - [x] Back Forward
  
- [ ] Model
  - [x] Initialize parameters
  - [x] Feed Forward
  - [x] Back Forward
  - [x] Train
    - [x] Calculate error
    - [x] Update parameters
    - [x] Predict
  
  - [ ] Batch
  - [ ] Early stop
  
- [x] Main
  - [x] Read data
  - [x] Train model
  - [x] Predict
  - [x] Write predictions
  
- [x] RUN
  - [x] Add modules
  - [x] Makefile
