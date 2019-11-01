//
// Created by Nameer Hirschkind on 2019-02-25.
//

#ifndef FIRSTLIBRARY_MODEL_H
#define FIRSTLIBRARY_MODEL_H
#include <iostream>
#include "Operation.h"
#include "Loss.h"

class Model {
    //private variables
    std::vector<Operation*> operations;
    Loss* lossFunction;
public:

    Model() {srand((unsigned int) (time(0)));}
    //functions
    void addOperation(Operation* op){
        operations.push_back(op);
    }

    void addInput(int size) {
        auto input = new Activation("none");
        input->initialize(size);
        operations.insert(operations.begin(),input);
    }

    void print() {
        for (int i=0;i<operations.size();i++){
            operations[i]->print();
        }
    }

    void show_storage() {
        for (int i=0;i<operations.size();i++){
            operations[i]->show_storage();
    }}

    MatrixXd forward_prop(MatrixXd X);

    void initialize();

    void clearStorage() {
        for (int i=0;i<operations.size();i++) {
            operations[i]->clearStorage();
        }
    }

    MatrixXd predict(MatrixXd x) {
        MatrixXd out = this->forward_prop(x);
        this -> clearStorage();
        return out;
    }

    void setLoss(std::string loss) {
        lossFunction = new Loss(loss);
    }

    double computeLoss(MatrixXd y);

    void concatAll();

    void updateAll(MatrixXd y,double lrate);

    void showWeights() {
        for (int i=0;i<operations.size();i++) {
            operations[i]->showW();
        }
    }

    void fit(MatrixXd x, MatrixXd y,double lrate,int batch_size,unsigned int batches,int displayLoss);
};


#endif //FIRSTLIBRARY_MODEL_H
