//
// Created by Nameer Hirschkind on 2019-02-25.
//

#include "Operation.h"



//Operation:
MatrixXd Operation::forward_prop(MatrixXd input) {
    return input;
}

void Operation::initialize(int insize) {
    in_size = insize;
}

MatrixXd Operation::update(MatrixXd nextGrad,int ind,Operation* prev,double lrate) {
    return nextGrad;
}

void Operation::concatStorage() {
    MatrixXd concated = concatVector(storage);
    this->clearStorage();
    storage = std::vector<MatrixXd>({concated});
}

//LinearTransformation

MatrixXd LinearTransformation::forward_prop(MatrixXd input) {
    MatrixXd output= W*input;
    storage.push_back(output);
    return output;
}
void LinearTransformation::initialize(int insize) {
    in_size = insize;
    W = MatrixXd::Random(out_size,in_size);
}

MatrixXd LinearTransformation::update(MatrixXd nextGrad, int ind,Operation* prev,double lrate) {
    MatrixXd chunk = prev->getStorage()[0].block(0,ind,in_size,1);
    //std::cout << "storage pull"<<chunk << std::endl;
    //std::cout << "previous gradient"<<nextGrad<<std::endl;
    //std::cout << "W" << W<<std::endl;
    MatrixXd pass_grad = W.transpose()*nextGrad;
    MatrixXd wgrad = nextGrad*chunk.transpose();
    //std::cout << wgrad << std::endl;
    W.noalias()-=wgrad*lrate;
    return pass_grad;
}



double relu(double x) {
    if (x>0) {return x;}
    else {return 0;}
}

double sigmoid(double x) {
    return 1/(1+std::exp(-x));
}

double reluPrime(double x) {
    if (x>0) {return 1;}
    else {return 0;}
}
MatrixXd concatVector(std::vector<MatrixXd> in) {
    if (in.size()==1){
        return in[0];
    }
    else {
        MatrixXd concat = in[0];
        MatrixXd* ptr = &concat;
        for (int i = 1; i < in.size(); i++) {
            MatrixXd C(in[i].rows(),in[i].cols()+concat.cols());
            C << in[i],concat;
            *ptr = C;
        }
        return concat;
    }
}


//Activation
MatrixXd Activation::forward_prop(MatrixXd input) {
    MatrixXd output;
    if (type=="sigmoid") {
        output =  input.unaryExpr(&sigmoid);
    }

    else if (type=="relu") {output =  input.unaryExpr(&relu);
    }

    if (type=="none") {output = input;}

    storage.push_back(output);
    return output;
}

MatrixXd Activation::update(MatrixXd nextGrad, int ind,Operation* prev,double lrate) {
    MatrixXd chunk = storage[0].block(0,ind,out_size,1);
    if (type=="sigmoid") {
        MatrixXd grad = chunk.unaryExpr(&oneminus).array()*chunk.array();
        return grad.array()*nextGrad.array();
    }
    else if (type=="relu") {
        return chunk.unaryExpr(&reluPrime).array()*nextGrad.array();
    }
    else if (type=="none") {
        return nextGrad;
    }
}


//bias

void Bias::initialize(int insize) {
    in_size=insize;
    out_size=insize;
    b = MatrixXd::Zero(insize,1);
}

MatrixXd Bias::forward_prop(MatrixXd input) {
    MatrixXd out = MatrixXd(input.rows(),input.cols());
    for (int rind = 0;rind< input.rows();rind++) {
        for (int colind = 0;colind<input.cols();colind++) {
            out(rind,colind) = input(rind,colind)+b(rind,0);
        }
    }
    storage.push_back(out);
    return out;
}

MatrixXd Bias::update(MatrixXd nextGrad,int ind,Operation* prev,double lrate) {
    b-=nextGrad*lrate;
    return nextGrad;
}