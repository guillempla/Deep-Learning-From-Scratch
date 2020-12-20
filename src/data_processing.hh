#ifndef DATA_PROCESSING_HH
#define DATA_PROCESSING_HH

#include <iostream>
#include <vector>
#include <math.h>

using namespace std;

class Data_processing {
    private:
        string path;

    public:
        //___________CONSTRUCTORS__________
        Data_processing();
        Data_processing(const string& path);

        //___________SETTERS__________


        //___________GETTERS__________

        double read_csv(const string& file_name) const;
        double write_csv(const string& file_name) const;

};

#endif
