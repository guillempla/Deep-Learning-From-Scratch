CC    := g++
EXECS := activation_functions loss_functions matrix

main: main.cc activation_functions.cc loss_functions.cc matrix.cc
	$(CC) -O2 $+ -o $@

activation_functions: activation_functions.cc
	$(CC) $+ -o $@

loss_functions: loss_functions.cc
	$(CC) $+ -o $@

matrix: matrix.cc
	$(CC) $+ -o $@

clean:
	rm -fv $(EXECS) *.txt *.in
