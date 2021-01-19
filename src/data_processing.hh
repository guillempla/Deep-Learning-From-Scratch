#ifndef DATA_PROCESSING_HH
#define DATA_PROCESSING_HH

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept> // std::runtime_error
#include "matrix.hh"

using namespace std;

class Data_processing {
    private:
        string test_labels_path;
        string test_vectors_path;
        string train_labels_path;
        string train_vectors_path;
        string output_path;

        static unsigned count_lines(const string& file_name) ;
        static unsigned count_cols(const string& file_name) ;
        static unsigned count_labels(const string& file_name) ;

        static Matrix read_labels(const string& file_name) ;
        static Matrix read_vectors(const string& file_name) ;


    public:
        //___________CONSTRUCTORS__________
        Data_processing();
        Data_processing(const string& test_labels_path, const string& test_vectors_path,
                        const string& train_labels_path, const string& train_vectors_path,
                        const string& output_path);

        //___________SETTERS__________


        //___________GETTERS__________
        Matrix read_test_labels() const;
        Matrix read_train_labels() const;
        Matrix read_test_vectors() const;
        Matrix read_train_vectors() const;

        static Matrix convert_binary_matrix(Matrix& predictions);
        void write_predictions(Matrix& predictions) const;

};

#endif
