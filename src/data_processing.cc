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
    return this->read_labels(this->test_labels_path);
}

vector<int> Data_processing::read_train_labels() const {
    return this->read_labels(this->train_labels_path);
}

Matrix Data_processing::read_test_vectors() const {
    return this->read_vectors(this->test_vectors_path);
}

Matrix Data_processing::read_train_vectors() const {
    return this->read_vectors(this->train_vectors_path);
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

vector<int> Data_processing::read_labels(const string& file_name) const {
    vector<int> labels_vector;

    // Create an input filestream
    ifstream myFile(file_name);

    // Make sure the file is open
    if (!myFile.is_open()) throw runtime_error("Could not open file");

    string value;

    // Read data, line by line
    while (getline(myFile, value)) {
        labels_vector.push_back(stoi(value));
    }

    // Close file
    myFile.close();

    return labels_vector;
}

Matrix Data_processing::read_vectors(const string& file_name) const {
    unsigned num_rows = this->count_lines(file_name);
    unsigned num_cols = this->count_cols(file_name);

    Matrix matrix(num_rows+1, num_cols+1, 0);

    // Create an input filestream
    ifstream myFile(file_name);

    // Make sure the file is open
    if (!myFile.is_open()) throw runtime_error("Could not open file");

    int i = 0;
    string line;
    while (getline(myFile, line)) {
        int j = 0;
        double value;
        stringstream ss(line);
        while (ss >> value) {
            matrix(i,j) = value;

            // If the next token is a comma, ignore it and move on
            if(ss.peek() == ',') ss.ignore();

            j++;
        }
        i++;
    }

    return matrix;
}
