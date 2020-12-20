#include "data_processing.hh"

//___________CONSTRUCTORS__________
Data_processing::Data_processing() {
    this->path = "./";
}

Data_processing::Data_processing(const string& path) {
    this->path = path;
}

//___________SETTERS__________


//___________GETTERS__________
double Data_processing::read_csv(const string& file_name) const {}

double Data_processing::write_csv(const string& file_name) const {}
