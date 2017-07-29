#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

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
  P_ = F_ * P_ * F_.transpose() + Q_;

}

void KalmanFilter::Update(const VectorXd &z) {
  
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;

  UpdateHelper(y);

}

void KalmanFilter::UpdateEKF(const VectorXd &z) {

  double rho = sqrt( pow( x_(0), 2 ) + pow( x_(1), 2 ) );
  double eta = atan( x_(1) / x_(0) );
  double rho_d = ( (x_(0)*x_(2) + x_(1)*x_(3)) / 
                    sqrt( pow( x_(0), 2 ) + pow( x_(1), 2 )) );

  VectorXd h(3);
  h << rho, eta, rho_d;
  VectorXd y = z - h;

  UpdateHelper(y);

}

void KalmanFilter::UpdateHelper(const VectorXd &y) {

  MatrixXd S = H_ * P_ * H_.transpose() + R_;
  MatrixXd K = P_ * H_.transpose() * S.inverse();
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());

  x_ = x_ + ( K * y );
  P_ = ( I - (K * y) ) * P_; 

}

