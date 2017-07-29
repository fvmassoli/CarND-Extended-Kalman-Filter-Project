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

  if(estimations.size() != 0 && estimations.size() == ground_truth.size()) {

      VectorXd tmp(4);
      int estimations_size = estimations.size();

      for(int i=0; i<estimations_size; i++) {

          tmp = estimations[i] - ground_truth[i];
          tmp = tmp.array() * tmp.array();

          rmse += tmp;
      }

    rmse = rmse  / estimations_size;
    rmse = rmse.array().sqrt();

  }

  return rmse;

}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {

  MatrixXd Hj(3, 4);
  double epsilon = 0.0001;

  double p_x = x_state(0);
  double p_y = x_state(1);
  double v_x = x_state(2);
  double v_y = x_state(3);

  double e1 = ( p_x * p_x ) + ( p_y * p_y );
  double e2 = sqrt(e1);
  double e3 = ( e1 * e2 );

  if( fabs(e1) < epsilon) {
    cout << "CalculateJacobian() error. Trying to divide by 0" << endl;
    return Hj;
  }
    
  Hj << (p_x/e2), (p_y/e2), 0, 0,
		   -(p_y/e1), (p_x/e1), 0, 0,
		     p_y*(v_x*p_y - v_y*p_x)/e3, p_x*(p_x*v_y - p_y*v_x)/e3, p_x/e2, p_y/e2;

	return Hj;

}
