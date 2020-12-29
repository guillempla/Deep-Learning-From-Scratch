#ifndef MODEL_HH
#define MODEL_HH

#include <iostream>
#include <vector>
#include <math.h>

using namespace std;

class Model {
    private:
        vector<Layer> layers;

    public:
        //___________CONSTRUCTORS__________
        Model (const vector<unsigned>& arquitecture);

        //___________SETTERS__________
        void feedForward(const vector<double>& inputs);
        void backPropagate(const vector<double>& predictions);



};

#endif
