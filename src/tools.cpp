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
  rmse << 0, 0, 0, 0;

  if(estimations.size() != ground_truth.size() || estimations.size() == 0){
    cout << "Invalid data. Cannot evaluate RMSE" << endl;
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
  rmse = rmse / estimations.size();

  //calculate the squared root
  rmse = rmse.array().sqrt();

  //return the result
return rmse;

}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {

  MatrixXd Hj(3, 4);
  Hj << 0,0,0,0,
        0,0,0,0,
        0,0,0,0;

  float p_x = x_state(0);
  float p_y = x_state(1);
  float v_x = x_state(2);
  float v_y = x_state(3);

  float e1 = ( p_x * p_x ) + ( p_y * p_y );

  if(fabs(e1) < 0.0001){
    std::cout << "Function CalculateJacobian() has Error: Division by Zero" << std::endl;
    return Hj;
}

  float e2 = sqrt(e1);
  float e3 = ( e1 * e2 );
    
  Hj << (p_x/e2), (p_y/e2), 0, 0,
		   -(p_y/e1), (p_x/e1), 0, 0,
		     p_y*(v_x*p_y - v_y*p_x)/e3, p_x*(p_x*v_y - p_y*v_x)/e3, p_x/e2, p_y/e2;

	return Hj;

}
