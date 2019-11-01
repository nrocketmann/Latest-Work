//
// Created by Nameer Hirschkind on 2019-02-25.
//

#ifndef FIRSTLIBRARY_OPERATION_H
#define FIRSTLIBRARY_OPERATION_H
#include <vector>
#include <iostream>
#include "Eigen/Eigen/Dense"
#include "Loss.h"


using Eigen::MatrixXd;
using Eigen::VectorXd;



class Operation {
protected:
    int in_size;
    int out_size;
    std::vector<MatrixXd> storage;
public:
    virtual void initialize(int insize);
    virtual MatrixXd forward_prop (MatrixXd input);
    void clearStorage() {storage.clear();};
    virtual void print() {std::cout << "base operation" <<std::endl;};
    void show_storage() {
        for (int i = 0;i<storage.size();i++) {
            std::cout << storage[i] << std::endl;
        }
    }
    int getInSize() {return in_size;}
    int getOutSize() {return out_size;}
    std::vector<MatrixXd> getStorage() {return storage;}
    virtual MatrixXd update(MatrixXd nextGrad,int ind, Operation* prev,double lrate);
    void concatStorage();
    virtual void showW() {}
};




class LinearTransformation: public Operation {
    MatrixXd W;

public:
    LinearTransformation(int outsize) { out_size = outsize;}
    void initialize(int insize) override;
    MatrixXd forward_prop (MatrixXd input) override;
    void print() override {
        std::cout << "linear transformation of size " << out_size <<std::endl;
    }
    MatrixXd update(MatrixXd nextgrad, int ind, Operation* prev,double lrate) override;
    void showW() override {std::cout << W << std::endl;}
};


class Activation: public Operation {
    std::string type; //options: sigmoid, relu
public:
    Activation(std::string s) {type=s;}
    MatrixXd forward_prop(MatrixXd input) override;
    void print() override {
        std::cout << "activation of type "<< type << " and size " << in_size<<std::endl;
    }
    void initialize(int insize) override {
        in_size=insize;
        out_size=insize;
    }
    MatrixXd update(MatrixXd nextGrad,int ind,Operation* prev,double lrate) override;

};

MatrixXd concatVector(std::vector<MatrixXd> in);


class Bias: public Operation {
    MatrixXd b;
public:
void initialize(int insize) override;
void print() override {std::cout << "bias added here"<<std::endl;};
MatrixXd forward_prop(MatrixXd input) override;
MatrixXd update(MatrixXd nextGrad,int ind,Operation* prev,double lrate) override;
void showW() override {std::cout << b << std::endl;}

};

#endif //FIRSTLIBRARY_OPERATION_H
