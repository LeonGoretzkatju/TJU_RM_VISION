#pragma once

#include "opencv2/opencv.hpp"
#include "Eigen/Dense"

namespace EigenKalman{
class KalmanFilter
{
private:
    int stateSize;     //状态方程维数
    int measSize;      //测量参数个数
    Eigen::MatrixXd A; //状态转移矩阵
    Eigen::MatrixXd P; //协方差矩阵
    Eigen::MatrixXd H; //测量矩阵
    Eigen::MatrixXd R; //测量误差
    Eigen::MatrixXd Q; //预测误差
public:
    KalmanFilter() {
	}
    /**
    * @函数: init
    * @描述: 初始化kalman预判参数
        */
    void init(int state, int meas, Eigen::MatrixXd & _A, Eigen::MatrixXd & _P, Eigen::MatrixXd & _R, Eigen::MatrixXd & _Q, Eigen::MatrixXd & _H){
		stateSize = state;
		measSize = meas;
		A = _A;
		P = _P;
		R = _R;
		Q = _Q;
		H = _H;
	}
    void predict(Eigen::VectorXd &x,Eigen::MatrixXd &F){
		A = F;
		x = A * x;
		Eigen::MatrixXd A_T = A.transpose();
		P = A * P*A_T + Q;
	}

	void update(Eigen::VectorXd &x, Eigen::VectorXd z_meas){
		Eigen::MatrixXd temp1, temp2, Ht;
		Ht = H.transpose();
		temp1 = H * P * Ht + R;
		temp2 = temp1.inverse();//(H*P*H'+R)^(-1)
		Eigen::MatrixXd K = P * Ht*temp2;
		Eigen::VectorXd z = H * x;
		x = x + K * (z_meas - z);
		Eigen::MatrixXd I = Eigen::MatrixXd::Identity(stateSize, stateSize);
		P = (I - K * H)*P;
	}
};
}
