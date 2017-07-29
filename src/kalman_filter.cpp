#include "kalman_filter.h"
#include <iostream>

using namespace std;

using Eigen::MatrixXd;
using Eigen::VectorXd;

const float DoublePI = 2 * M_PI;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;

}

void KalmanFilter::Predict() {
  
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;

}

void KalmanFilter::Update(const VectorXd &z) {

  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;

  UpdateHelper(y);

}

void KalmanFilter::UpdateEKF(const VectorXd &z) {

  float p_x = x_(0);
  float p_y = x_(1);
  float v_x = x_(2);
  float v_y = x_(3);
    
  float rho = sqrt( (p_x * p_x) + (p_y * p_y) );
  
  if(rho < 0.00001) {
    rho = 0.00001;
  }

  float phi = atan2(p_y, p_x);
  float rho_d = ( (p_x * v_x) + (p_y * v_y) ) / rho;
    
  VectorXd h(3);
  h << rho, phi, rho_d;
  VectorXd y = z - h;

  while(y(1) > M_PI){
    y(1) -= DoublePI;
  }

  while(y(1) < -M_PI){
    y(1) += DoublePI;
  }

  UpdateHelper(y);

}

void KalmanFilter::UpdateHelper(const VectorXd &y) {
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * Ht * Si;
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

