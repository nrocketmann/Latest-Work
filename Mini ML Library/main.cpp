#include <iostream>
#include "Model.h"

#include "Operation.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

//X DATA MUST BE IN COLUMN VECTORS
//Y DATA MUST BE A SINGLE ROW VECTOR

int main() {


    auto model = Model();

    auto l1 = new LinearTransformation(2);
    auto b1 = new Bias();
    auto a1 = new Activation("relu");
    auto l2 = new LinearTransformation(2);
    auto b2 = new Bias();
    auto a2 = new Activation("sigmoid");
    auto l3 = new LinearTransformation(1);
    auto b3 = new Bias();
    auto a3 = new Activation("sigmoid");


    model.addInput(2);
    model.addOperation(l1);
    model.addOperation(b1);
    model.addOperation(a1);
    model.addOperation(l2);
    model.addOperation(b2);
    model.addOperation(a2);
    model.addOperation(l3);
    model.addOperation(b3);
    //model.addOperation(a3);
    model.setLoss("mse");


    model.initialize();
    model.print();
    MatrixXd test(2,4);
    test <<1,0,1,0,
    1,0,0,1;

    auto labels = MatrixXd(1, 4);
    labels << 2, 0, 1, 1;
    model.fit(test,labels,.01,4,100000,10000);
    //model.fit(test,labels,.01,4,50000,10000);
    cout << model.predict(test) << endl;


    return 0;
}