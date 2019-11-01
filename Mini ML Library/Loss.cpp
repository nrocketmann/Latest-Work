//
// Created by Nameer Hirschkind on 2019-03-01.
//

#include "Loss.h"

double square(double n) {return n*n;}
double crossentropy(double p,double y) {
    if (y == 1) { return -log(p); }
    else if (y == 0) { return -log(1 - p); }
    else {
        std::cout << "Bad y input" << std::endl;
        return 0;
    }
}

double special_log1(double n) {return -log(n);}
double special_log2(double n) {return -log(1-n);}
double oneminus(double n) {return 1-n;}
double times(double n) {return n*2;}

MatrixXd Loss::compute_loss(MatrixXd preds, MatrixXd y) {
    if (type=="mse") {
        //std::cout <<preds<<std::endl;
        //std::cout<<y<<std::endl;
        MatrixXd squared = (preds-y).unaryExpr(&square);
        return squared;
    }
    else if (type=="crossentropy") {
        MatrixXd part1 = preds.unaryExpr(&special_log1).array()*y.array();
        MatrixXd part2 = y.unaryExpr(&oneminus).array()*preds.unaryExpr(&special_log2).array();
        return part1+part2;
    }
    else {
        std::cout <<"you done goofed the loss function"<<std::endl;

    }
}
MatrixXd Loss::loss_grad(const MatrixXd preds,const MatrixXd y) {
    if (type=="mse") {
        return (preds-y).unaryExpr(&times);
    }
    else {std::cout<<"YOU CAN ONLY USE MSE"<<std::endl;}
}

