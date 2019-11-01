//
// Created by Nameer Hirschkind on 2019-03-01.
//

#ifndef FIRSTLIBRARY_LOSS_H
#define FIRSTLIBRARY_LOSS_H
#include <iostream>
#include "Eigen/Eigen/Dense"
using Eigen::MatrixXd;
using Eigen::ArrayXXf;

class Loss {
    std::string type;

public:
    Loss(std::string s) {type = s;}
    MatrixXd compute_loss(MatrixXd preds, MatrixXd y);
    MatrixXd loss_grad(const MatrixXd preds,const MatrixXd y);


};

double times(double n);
double special_log1(double n);
double special_log2(double n);
double oneminus(double n);


#endif //FIRSTLIBRARY_LOSS_H
