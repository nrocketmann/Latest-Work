//
// Created by Nameer Hirschkind on 2019-02-25.
//

#include "Model.h"

MatrixXd Model::forward_prop(MatrixXd X) {
MatrixXd* storage = &X;
for (int i=0; i<operations.size();i++) {
    *storage = operations[i]->forward_prop(*storage);

}
return *storage;
}

void Model::initialize() {
    std::cout << operations.size() <<std::endl;
    for (int i = 1;i<operations.size();i++) {
        operations[i]->initialize(operations[i-1]->getOutSize());
    }
}



double Model::computeLoss(MatrixXd y) {
    MatrixXd joined = concatVector(operations[operations.size()-1]->getStorage());
    MatrixXd losses = lossFunction->compute_loss(joined,y);
    return losses.sum()/losses.cols();
}

void Model::concatAll() {
    for (int i=0;i<operations.size();i++) {
        operations[i]->concatStorage();
    }
}

void Model::updateAll(MatrixXd y, double lrate) {
    this->concatAll();
    MatrixXd preds = operations[operations.size()-1]->getStorage()[0];
    if (y.cols()!=preds.cols()) {
        std::cout << "WRONG DIMS FOR LABELS"<<std::endl;
    }
    MatrixXd lossGrads = lossFunction->loss_grad(preds,y);
    for (int ind = 0; ind<y.cols();ind++) { // for each data point
        MatrixXd lgrad = lossGrads.block(0,ind,1,1);
        MatrixXd* grad_store = &lgrad;
        for (int layer = operations.size()-1;layer>0;layer--) {
            MatrixXd pass_grad = operations[layer]->update(*grad_store,ind,operations[layer-1],lrate);
            *grad_store = pass_grad;
        }
    }
}

void Model::fit(MatrixXd x, MatrixXd y, double lrate, int batch_size,unsigned int batches,int displayLoss) {
    double loss = 0;
    double* lossptr = &loss;
    for (unsigned int iter=0; iter<batches;iter++) {
        std::vector<MatrixXd> allx;
        std::vector<MatrixXd> ally;
        for (int i = 0; i < batch_size; i++) {
            int ind = rand() % x.cols();
            allx.emplace_back(x.block(0,ind, x.rows(), 1));
            ally.emplace_back(y.block(0,ind,1,1));
        };
        MatrixXd input = concatVector(allx);
        MatrixXd labels = concatVector(ally);
        //std::cout << input << std::endl;
        //std::cout << labels << std::endl;
        this->forward_prop(input);
        *lossptr+=this->computeLoss(labels);
        this->updateAll(labels,lrate);
        this->clearStorage();
        if (iter%displayLoss==0 && iter!=0) {
            std::cout << "Average loss at batch "<< iter << ": " << *lossptr/displayLoss << std::endl;
            *lossptr = 0;
        }
    }

}