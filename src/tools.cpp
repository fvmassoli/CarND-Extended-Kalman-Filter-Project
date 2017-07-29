#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  if(estimations.size() != ground_truth.size() || estimations.size() == 0){
    cout << "Invalid input. Cannot evaluate RMSE!!" << endl;
    return rmse;
  }

  //accumulate squared residuals
  for(unsigned int i=0; i < estimations.size(); ++i){

    VectorXd tmp = estimations[i] - ground_truth[i];

    //coefficient-wise multiplication
    tmp = tmp.array()*tmp.array();
    rmse += tmp;
  }

  //calculate the mean
  rmse = rmse/estimations.size();

  //calculate the squared root
  rmse = rmse.array().sqrt();

  //return the result
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
  TODO:
    * Calculate a Jacobian here.
  */
  MatrixXd Hj(3,4);
  Hj << 0,0,0,0,
        0,0,0,0,
        0,0,0,0;

  //recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  //pre-compute a set of terms to avoid repeated calculation
  float e1 = px*px+py*py;
  float e2 = sqrt(e1);
  float e3 = (e1*e2);

  //check division by zero
  if(fabs(e1) < 0.0001){
    std::cout << "Cannot divide by 0 for Jacobian" << std::endl;
    return Hj;
  }

  //compute the Jacobian matrix
  Hj << (px/e2),                (py/e2),                0,      0,
        -(py/e1),               (px/e1),                0,      0, 
        py*(vx*py - vy*px)/e3,  px*(px*vy - py*vx)/e3,  px/e2,  py/e2;

  return Hj;

}
