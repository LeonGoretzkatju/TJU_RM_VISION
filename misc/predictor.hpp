/*
 * 这是 TJU Robomasters 上位机源码，未经管理层允许严禁传播给其他人（包括队内以及队外）
 *
 * 该文件包含各种预测模型，并封装到了以抽象类Predictor类为基类的类中
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "util.hpp"
#include <atomic>
#include "KalmanFilter.hpp"
using namespace cv;

typedef struct _prd_path_pt{

    Point3f worldPosition;
    Point2f targetPTZAngle;
    Point2f selfWorldPosition;
    double duration;

    _prd_path_pt(Point3d _worldp,Point2f _tgtAngle,Point2f _sfworldp,float _dur)
    {
        worldPosition = _worldp;
        targetPTZAngle = _tgtAngle;
        selfWorldPosition = _sfworldp;
        duration = _dur;
    }

    _prd_path_pt()
    { }

}PredictionPathPoint;


class Predictor
{
public:
    // 添加一个轨迹点，该函数应该能够自主甄别轨迹点是否与之前追踪的目标一样，如果不是
    // 预测器自动清空历史数据重新开始预测
    virtual void AddPredictPoint(PredictionPathPoint ppp) = 0;

    // 预测一段时间之后目标的位置
    virtual void Predict(double prdTime) = 0;
    
    // 主动清除历史记录
    virtual void ClearHistory() = 0;
};
// 线性预测器，由历史数据给出线性的预测（假设目标匀速直线运动）
class KFCPredictor : public Predictor
{
public:
    static const int HistorySize = 6;
    Mat P, Q, H, R, F, K, x_;
    int xstate = 0;
    PredictionPathPoint present;
    PredictionPathPoint lastpoint;
    
    KFCPredictor()
    {
        Mat P_in = Mat::eye(6,6,CV_32FC1);
        Mat Q_in = Mat::eye(6,6,CV_32FC1);
        Mat H_in = Mat::eye(6,6,CV_32FC1);
        Mat R_in = Mat::eye(6,6,CV_32FC1);
        P = P_in;
        Q = Q_in;
        H = H_in;
        R = R_in;
        lastpoint.targetPTZAngle=Point2f(0,0);
    }
    
    KFCPredictor(Mat P_in, Mat Q_in, Mat H_in, Mat R_in)
    {
        P = P_in;
        Q = Q_in;
        H = H_in;
        R = R_in;
        lastpoint.targetPTZAngle=Point2f(0,0);
    }
    
    void AddPredictPoint(PredictionPathPoint ppp)
    {
        if (Length(ppp.targetPTZAngle - lastpoint.targetPTZAngle) > 10)
            ClearHistory();
        present = ppp;
    }
    
    void ClearHistory()
    {
        xstate = 0;
        P = Mat::eye(6,6,CV_32FC1);
    };

    void FirstFind()
    {
        p_tx_old = present.worldPosition.x;
        p_ty_old = present.worldPosition.y;
        p_tz_old = present.worldPosition.z;
        lastpoint = present;
    }

    void FirstSetFilter()
    {
        //将视觉根据图像计算出的装甲板在相机坐标系下的位置传给预测类中的worldposition 
        //以当前的计算作为根据上一次计算进行预测的观测值
        double t = present.duration;
        float v_tx_now = (present.worldPosition.x - p_tx_old)/t;
        float v_ty_now =(present.worldPosition.y - p_ty_old)/t;
        float v_tz_now = (present.worldPosition.z - p_tz_old)/t;
        x_ = (Mat_<float>(6,1) <<
                   present.worldPosition.x, present.worldPosition.y, present.worldPosition.z,
                   v_tx_now, v_ty_now, v_tz_now
                   );
        xstate = 1;
        p_tx_old = present.worldPosition.x;
        p_ty_old = present.worldPosition.y;
        p_tz_old = present.worldPosition.z;
        lastpoint = present;
    }
    void ContinueSetFilter()
    {
        double t = present.duration;
        float v_tx_now = (present.worldPosition.x - p_tx_old)/t;
        float v_ty_now = (present.worldPosition.y - p_ty_old)/t;
        float v_tz_now = (present.worldPosition.z - p_tz_old)/t;

        Mat z = (Mat_<float>(6,1) <<
                   present.worldPosition.x, present.worldPosition.y, present.worldPosition.z,
                   v_tx_now, v_ty_now, v_tz_now
                   );

        Predict(t);
        update(z);

        p_tx_old = x_.at<float>(0, 0);
        p_ty_old = x_.at<float>(1, 0);
        p_tz_old = x_.at<float>(2, 0);
        lastpoint = present;
    }

    void Predict(double prdTime)
    {
        double t = prdTime;
        F = (Mat_<float>(6,6) <<
                   1.0, 0.0, 0.0, t, 0.0, 0.0,
                   0.0, 1.0, 0.0, 0.0, t, 0.0,
                   0.0, 0.0, 1.0, 0.0, 0.0, t,
                   0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 1.0,0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 1.0
                   );
        //起
        x_ = F * x_;
        // return Point3f(x_.at<float>(0,0),x_.at<float>(0,1),x_.at<float>(0,2));
    }

    void update(Mat z)
    {
        //按
        P = F*P*F.t() + Q;
        //顿
        Mat S = H*P*H.t() + R; 
        K = P*H.t()*S.inv();
        //挫
        Mat y = z - H*x_;
        x_ = x_ + (K*y);
        //下笔风雷
        Mat I = Mat::eye(6,6,CV_32FC1);
        P = (I - K*H)*P;
    }
protected:
    float p_tx_old;                  //位置保留量
    float p_ty_old;
    float p_tz_old;
};

class AnglePredictor : public Predictor
{
public:
    
    Mat P, Q, H, R, F, K, x_;
    int xstate = 0;
    PredictionPathPoint present;
    PredictionPathPoint lastpoint;
    
    AnglePredictor()
    {
        Mat P_in = Mat::eye(4,4,CV_32FC1);
        Mat Q_in = (Mat_<float>(4,4) <<
                   5,0,0,0,
                   0,1,0,0,
                   0,0,5,0,
                   0,0,0,1
                   );
        Mat H_in = (Mat_<float>(2,4) <<
                   1,0,0,0,
                   0,1,0,0
                   );
        Mat R_in = (Mat_<float>(2,2) <<
                   200,0,
                   0,200
                   );
        P = P_in;
        Q = Q_in;
        H = H_in;
        R = R_in;
        lastpoint.targetPTZAngle=Point2f(0,0);
    }
    
    AnglePredictor(Mat P_in, Mat Q_in, Mat H_in, Mat R_in)
    {
        P = P_in;
        Q = Q_in;
        H = H_in;
        R = R_in;
        lastpoint.targetPTZAngle=Point2f(0,0);
    }
    
    void AddPredictPoint(PredictionPathPoint ppp)
    {
        if (Length(ppp.targetPTZAngle - lastpoint.targetPTZAngle) > 10)
            ClearHistory();
        present = ppp;
    }
    
    void ClearHistory()
    {
        xstate = 0;
        P = Mat::eye(4,4,CV_32FC1);
    };

    void FirstFind()
    {
        
        lastpoint = present;
    }

    void FirstSetFilter()
    {
        double t = present.duration;
        float vx = (present.targetPTZAngle.x - lastpoint.targetPTZAngle.x)/t;
        float vy =(present.targetPTZAngle.y - lastpoint.targetPTZAngle.y)/t;
        
        x_ = (Mat_<float>(4,1) <<
                   present.targetPTZAngle.x, present.targetPTZAngle.y,
                   vx, vy
                   );
        xstate = 1;
        
        lastpoint = present;
    }
    void ContinueSetFilter()
    {
        double t = present.duration;
        float vx = (present.targetPTZAngle.x - lastpoint.targetPTZAngle.x)/t;
        float vy =(present.targetPTZAngle.y - lastpoint.targetPTZAngle.y)/t;
        Mat z = (Mat_<float>(2,1) <<
                   present.targetPTZAngle.x, present.targetPTZAngle.y
                   );

        Predict(t);
        update(z);
        // cout << "presemtangle: " << present.targetPTZAngle << " lastangle: " << lastpoint.targetPTZAngle << endl;
        // cout << "speedx: " << (present.targetPTZAngle.x-lastpoint.targetPTZAngle.x)/present.duration << endl;
        // cout << "speedy: " << (present.targetPTZAngle.y-lastpoint.targetPTZAngle.y)/present.duration << endl;
        lastpoint = present;
    }

    void Predict(double prdTime)
    {
        double t = prdTime;
        F = (Mat_<float>(4,4) <<
                   1.0, 0.0, t, 0.0, 
                   0.0, 1.0, 0.0, t, 
                   0.0, 0.0, 1.0, 0.0, 
                   0.0, 0.0, 0.0, 1.0
                   );
        //起
        x_ = F * x_;
        
        //return Point2f(myx_.at<float>(0,0),myx_.at<float>(0,1));
    }
    Point2f PredictReal(double prdTime)
    {
        double t = prdTime;
        F = (Mat_<float>(4,4) <<
                   1.0, 0.0, t, 0.0, 
                   0.0, 1.0, 0.0, t, 
                   0.0, 0.0, 1.0, 0.0, 
                   0.0, 0.0, 0.0, 1.0
                   );
        //起
        
        Mat myx_ = F * x_;
        return Point2f(myx_.at<float>(0,0),myx_.at<float>(0,1));
    }

    void update(Mat z)
    {
        //按
        P = F*P*F.t() + Q;
        //顿
        Mat S = H*P*H.t() + R; 
        K = P*H.t()*S.inv();
        //挫
        Mat y = z - H*x_;
        x_ = x_ + (K*y);
        //下笔风雷
        Mat I = Mat::eye(4,4,CV_32FC1);
        P = (I - K*H)*P;
    }
};

class KalmanPredictor{
public:
KalmanPredictor(){

	Eigen::MatrixXd A(stateSize, stateSize);
	A << 1, 0, 1, 0,
		0, 1, 0, 1,
		0, 0, 1, 0,
		0, 0, 0, 1 ;

	Eigen::MatrixXd H(measureSize, stateSize);
	H << 2, 0, 0, 0,
		0, 2, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1;

	Eigen::MatrixXd P(stateSize, stateSize);
	P << 1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1;

	Eigen::MatrixXd Q(stateSize, stateSize);
	Q << 1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1;

	Eigen::MatrixXd R(measureSize, measureSize);
	R << 900, 0, 0, 0,
		0, 900, 0, 0,
		0, 0, 700, 0,
		0, 0, 0, 700;

	KF.init(stateSize, measureSize, A, P, R, Q, H);

	x.resize(stateSize);
	x << 0, 0, 0, 0;
}
public:
	Eigen::MatrixXd F;
	Eigen::VectorXd x;
    cv::Point2f predict(float yaw, float pitch,float distance, float deltatime,int bulletSpeed) {	
	    return kalmanPredict(yaw, pitch, (double)deltatime,distance/bulletSpeed * 0.001);
	}

	cv::Point2f kalmanPredict(float yaw, float pitch, double deltatime,float time){
	if (abs(x(0) - yaw) > 200) {
		x(0) = (double)yaw;
		yawSpeed = 0;
		x(2) = yawSpeed;
	}
	if (abs(x(2) - pitch) > 200) {
		x(1) = (double)pitch;
		pitchSpeed = 0;
		x(3) = pitchSpeed;
	}

	yawSpeed = (yaw - lastYaw) / deltatime;
	pitchSpeed = (pitch - lastPitch) / deltatime;
	if(abs(yawSpeed) > 50){
		yawSpeed = 0;
	}
	if(abs(pitchSpeed > 20)){
		pitchSpeed = 0;
	}
	Eigen::VectorXd z(measureSize);
	Eigen::VectorXd output;
	F = Eigen::MatrixXd(4, 4);
	cout << "hello3"<<endl;
    F << 1.0,0.0,(double)deltatime,0.0,
        0.0,1.0,0.0,(double)deltatime,
        0.0,0.0,1.0,0.0,
        0.0,0.0,0.0,1.0;
	z << (double)yaw, (double)pitch, (double)yawSpeed, (double)pitchSpeed;
	KF.predict(x,F);
	KF.update(x, z);
	lastYaw = yaw;
	lastPitch = pitch;
	//cout << " X:" << x(0) << " v:" << x(2) << endl;
	double predictYaw = x(0) + (time + 0.35) * x(2); 
	double predictPitch = x(1) + (time + 0.35) * x(3);

	return cv::Point2f((float)predictYaw, (float)predictPitch);
}
private:
	EigenKalman::KalmanFilter KF;
	double lastYaw;
	double lastPitch;
	double yawSpeed = 0;
	double pitchSpeed = 0;
	int measureSize = 4;
	int stateSize = 4;
public:
	atomic<int> shootSpeed;//发射速度 单位;米/秒
};
