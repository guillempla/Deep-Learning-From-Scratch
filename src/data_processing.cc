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

    string value;

    // Read data, line by line
    while(getline(myFile, value)) {
        labels_vector.push_back(stoi(value));
    }

    // Close file
    myFile.close();

    return labels_vector;
}

vector<int> Data_processing::read_train_labels() const {
    vector<int> labels_vector;

    // Create an input filestream
    ifstream myFile(this->train_labels_path);

    // Make sure the file is open
    if (!myFile.is_open()) throw runtime_error("Could not open file");

    string value;

    // Read data, line by line
    while(getline(myFile, value)) {
        labels_vector.push_back(stoi(value));
    }

    // Close file
    myFile.close();

    return labels_vector;
}

Matrix Data_processing::read_test_vectors() const {
    Matrix m(3,2);

    // Create an input filestream
    ifstream myFile(this->test_vectors_path);

    // Make sure the file is open
    if (!myFile.is_open()) throw runtime_error("Could not open file");

    string line;

    // Read data, line by line
    while(getline(myFile, line)) {

    }

    // Close file
    myFile.close();

    return m;
}

Matrix Data_processing::read_train_vectors() const {
    unsigned num_rows = this->count_lines(this->train_vectors_path);
    unsigned num_cols = this->count_cols(this->train_vectors_path);

    Matrix m_matrix(num_rows, num_cols, 0);

    return m_matrix;
}

void Data_processing::write_predictions(const vector<int>& predictions) const {
    // Create an output filestream object
    ofstream myOutput(output_path);

    for (int i = 0; i < predictions.size(); i++) {
        myOutput << to_string(predictions[i]) << "\n";
    }

    // Close the file
    myOutput.close();
}


//___________PRIVATE__________
unsigned Data_processing::count_lines(const string& file_name) const {
    unsigned number_of_lines = 0;
    string line;
    ifstream myfile(file_name);
    if (!myfile.is_open()) throw runtime_error("Could not open file");

    while (getline(myfile, line))
        ++number_of_lines;

    myfile.close();
    return number_of_lines;
}

unsigned Data_processing::count_cols(const string& file_name) const {
    string line;
    ifstream myfile(file_name);
    if (!myfile.is_open()) throw runtime_error("Could not open file");

    getline(myfile, line);
    unsigned number_of_cols = 0;
    for (auto c: line)
        if (c == ',')
            number_of_cols++;

    myfile.close();
    return number_of_cols;
}
