CC    := g++
EXECS := data_processing activation_functions loss_functions matrix

main: main.cc data_processing activation_functions.cc loss_functions.cc matrix.cc
	$(CC) -O2 $+ -o $@

data_processing: data_processing.cc
	$(CC) $+ -o $@

activation_functions: activation_functions.cc
	$(CC) $+ -o $@

loss_functions: loss_functions.cc
	$(CC) $+ -o $@

matrix: matrix.cc
	$(CC) $+ -o $@

clean:
	rm -fv $(EXECS) *.txt *.in
