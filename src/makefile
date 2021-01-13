CC				:= g++
EXECS 			:= network data_processing activation loss matrix layer model
CFLAGS 			:= -fopenmp -ffp-contract=fast -funsafe-math-optimizations -march=native -msse2 -mfpmath=sse -ftree-vectorize -std=c++17
OPTIMIZATION	:= -Ofast


network: main.cc data_processing.cc activation.cc loss.cc matrix.cc layer.cc model.cc
	$(CC) $(CFLAGS) $(OPTIMIZATION) $+ -o $@

data_processing: data_processing.cc
	$(CC) $+ -o $@

activation: activation.cc
	$(CC) $+ -o $@

loss: loss.cc
	$(CC) $+ -o $@

matrix: matrix.cc
	$(CC) $+ -o $@

layer: layer.cc
	$(CC) $+ -o $@

model: model.cc
	$(CC) $+ -o $@



clean:
	rm -fv $(EXECS)
