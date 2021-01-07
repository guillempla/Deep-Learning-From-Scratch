CC    := g++
EXECS := data_processing activation loss_functions matrix layer model

main: main.cc data_processing.cc activation.cc loss_functions.cc matrix.cc layer.cc model.cc
	$(CC) -O2 $+ -o $@

data_processing: data_processing.cc
	$(CC) $+ -o $@

activation: activation.cc
	$(CC) $+ -o $@

loss_functions: loss_functions.cc
	$(CC) $+ -o $@

matrix: matrix.cc
	$(CC) $+ -o $@

layer: layer.cc
	$(CC) $+ -o $@

model: model.cc
	$(CC) $+ -o $@



clean:
	rm -fv $(EXECS) *.txt *.in
