#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "util.hpp"

using namespace cv;
using namespace std;
class CoordinatesFusion
{    
public:

    CoordinatesFusion(){
	
    }
    bool caculateTargetPose(cv::Point2f angle, cv::Mat tvecDeviation,  float distance, cv::Point2f &targetPose,ImageData &frame) {
	// GimblaPose gimblaPose;

	Mat tvec(1, 3, CV_32FC1);

	tvec.at<float>(0, 0) = (float)distance * tan(angle.x / 180 * CV_PI) + tvecDeviation.at<float>(0, 0);
	tvec.at<float>(0, 1) = (float)distance * tan(angle.y / 180 * CV_PI) + tvecDeviation.at<float>(0, 1);
	tvec.at<float>(0, 2) = (float)distance + tvecDeviation.at<float>(0, 2);

	Point2f correctedAngle;
	correctedAngle.x = atan2(tvec.at<float>(0, 0), tvec.at<float>(0, 2)) * 180 / CV_PI;
	correctedAngle.y = atan2(tvec.at<float>(0, 1), tvec.at<float>(0, 2)) * 180 / CV_PI;

	targetPose.x = frame.ptzAngle.x - correctedAngle.x ;

	targetPose.y = frame.ptzAngle.y - correctedAngle.x;
	if (targetPose.y > -5.0)
		targetPose.y = -5.0;
	else if(targetPose.y < -47.0)
		targetPose.y = -47.0;
	return true;
}


protected:

};
