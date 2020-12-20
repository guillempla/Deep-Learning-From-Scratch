#include "data_processing.hh"

//___________CONSTRUCTORS__________
Data_processing::Data_processing() {
    this->test_labels_path = "../data/fashion_mnist_test_labels.csv";
    this->test_vectors_path = "../data/fashion_mnist_test_vectors.csv";
    this->train_labels_path = "../data/fashion_mnist_train_labels.csv";
    this->train_vectors_path = "../data/fashion_mnist_train_vectors.csv";
    this->output_path = "../predictions";
}

Data_processing::Data_processing(const string& test_labels_path, const string& test_vectors_path, const string& train_labels_path, const string& train_vectors_path, const string& output_path) {
    this->test_labels_path = test_labels_path;
    this->test_vectors_path = test_vectors_path;
    this->train_labels_path = train_labels_path;
    this->train_vectors_path = train_vectors_path;
    this->output_path = output_path;
}

//___________SETTERS__________


//___________GETTERS__________
vector<int> Data_processing::read_test_labels() const {
    vector<int> labels_vector;

    // Create an input filestream
    ifstream myFile(this->test_labels_path);

    // Make sure the file is open
    if (!myFile.is_open()) throw runtime_error("Could not open file");

    // Read data, line by line
    while(getline(myFile, value)) {
        labels_vector.push_back(value);
    }

    // Close file
    myFile.close();

    return labels_vector;
}

Matrix Data_processing::read_test_vectors() const {
    Matrix matrix;

    ifstream myTestVectors(this->test_vectors_path);

    return matrix;
}

double Data_processing::write_predictions(const vector<int>& predictions) const {
    // Create an output filestream object
    ofstream myOutput(output_path);

    for (int i = 0; i < predictions.size(); i++) {
        myOutput << toString(predictions[i]) << "\n";
    }

    // Close the file
    myOutput.close();

    return 0;
}
