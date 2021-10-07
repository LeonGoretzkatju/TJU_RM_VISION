/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2013, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/
#pragma once

#include "custom_definations.hpp"
#include <algorithm>
#include <limits.h>

#if defined DEVICE_TX
#include "opencv2/imgproc.hpp"
#endif

namespace cv {
namespace tracking_internal
{

    inline double abs(double x)
    {
        return x >= 0 ? x : -x;
    }

/** Computes normalized corellation coefficient between the two patches (they should be
* of the same size).*/
    double computeNCC(const Mat& patch1, const Mat& patch2)
    {
        int N = patch1.rows * patch1.cols;

        if(N <= 1000 && patch1.type() == CV_8U && patch2.type() == CV_8U)
        {
            unsigned s1 = 0, s2 = 0;
            unsigned n1 = 0, n2 = 0;
            unsigned prod = 0;

            if(patch1.isContinuous() && patch2.isContinuous())
            {
                const uchar* p1Ptr = patch1.ptr<uchar>(0);
                const uchar* p2Ptr = patch2.ptr<uchar>(0);

                for(int j = 0; j < N; j++)
                {
                  s1 += p1Ptr[j];
                  s2 += p2Ptr[j];
                  n1 += p1Ptr[j]*p1Ptr[j];
                  n2 += p2Ptr[j]*p2Ptr[j];
                  prod += p1Ptr[j]*p2Ptr[j];
                }
            }
            else
            {
                for(int i = 0; i < patch1.rows; i++)
                {
                  const uchar* p1Ptr = patch1.ptr<uchar>(i);
                  const uchar* p2Ptr = patch2.ptr<uchar>(i);

                  for(int j = 0; j < patch1.cols; j++)
                  {
                    s1 += p1Ptr[j];
                    s2 += p2Ptr[j];
                    n1 += p1Ptr[j]*p1Ptr[j];
                    n2 += p2Ptr[j]*p2Ptr[j];
                    prod += p1Ptr[j]*p2Ptr[j];
                  }
                }
            }

            double sq1 = sqrt(std::max(0.0, n1 - 1.0 * s1 * s1 / N));
            double sq2 = sqrt(std::max(0.0, n2 - 1.0 * s2 * s2 / N));
            return (sq2 == 0) ? sq1 / abs(sq1) : (prod - 1.0 * s1 * s2 / N) / sq1 / sq2;
        }
        else
        {
            double s1 = sum(patch1)(0);
            double s2 = sum(patch2)(0);
            double n1 = norm(patch1, NORM_L2SQR);
            double n2 = norm(patch2, NORM_L2SQR);
            double prod=patch1.dot(patch2);
            double sq1 = sqrt(std::max(0.0, n1 - 1.0 * s1 * s1 / N));
            double sq2 = sqrt(std::max(0.0, n2 - 1.0 * s2 * s2 / N));
            return (sq2 == 0) ? sq1 / abs(sq1) : (prod - s1 * s2 / N) / sq1 / sq2;
        }
    }

    template<typename T>
    T getMedianAndDoPartition(std::vector<T>& values)
    {
        size_t size = values.size();
        if(size%2==0)
        {
            std::nth_element(values.begin(), values.begin() + size/2-1, values.end());
            T firstMedian = values[size/2-1];

            std::nth_element(values.begin(), values.begin() + size/2, values.end());
            T secondMedian = values[size/2];

            return (firstMedian + secondMedian) / (T)2;
        }
        else
        {
            size_t medianIndex = (size - 1) / 2;
            std::nth_element(values.begin(), values.begin() + medianIndex, values.end());

            return values[medianIndex];
        }
    }

    template<typename T>
    T getMedian(const std::vector<T>& values)
    {
        std::vector<T> copy(values);
        return getMedianAndDoPartition(copy);
    }
}
}



namespace cv
{

typedef Rect_<double> Rect2d;
typedef Rect_<int> Rect2i;

class CV_EXPORTS TrackerModel
{
 public:

  /**
   * \brief Constructor
   */
  TrackerModel(){}

  /**
   * \brief Destructor
   */
  virtual ~TrackerModel(){}

  /** @brief Set TrackerEstimator, return true if the tracker state estimator is added, false otherwise
    @param trackerStateEstimator The TrackerStateEstimator
    @note You can add only one TrackerStateEstimator
     */
  //bool setTrackerStateEstimator( Ptr<TrackerStateEstimator> trackerStateEstimator );

  /** @brief Estimate the most likely target location

    @cite AAM ME, Model Estimation table I
    @param responses Features extracted from TrackerFeatureSet
     */
  void modelEstimation( const std::vector<Mat>& responses );

  /** @brief Update the model

    @cite AAM MU, Model Update table I
     */
  void modelUpdate();

  /** @brief Run the TrackerStateEstimator, return true if is possible to estimate a new state, false otherwise
    */
  bool runStateEstimator();

  /** @brief Set the current TrackerTargetState in the Trajectory
    @param lastTargetState The current TrackerTargetState
     */
  //void setLastTargetState( const Ptr<TrackerTargetState>& lastTargetState );

  /** @brief Get the last TrackerTargetState from Trajectory
    */
  //Ptr<TrackerTargetState> getLastTargetState() const;

  /** @brief Get the list of the ConfidenceMap
    */
  //const std::vector<ConfidenceMap>& getConfidenceMaps() const;

  /** @brief Get the last ConfidenceMap for the current frame
     */
  //const ConfidenceMap& getLastConfidenceMap() const;

  /** @brief Get the TrackerStateEstimator
    */
  //Ptr<TrackerStateEstimator> getTrackerStateEstimator() const;

 private:

  //void clearCurrentConfidenceMap();

 protected:
  //std::vector<ConfidenceMap> confidenceMaps;
  //Ptr<TrackerStateEstimator> stateEstimator;
  //ConfidenceMap currentConfidenceMap;
  //Trajectory trajectory;
  int maxCMLength;

  virtual void modelEstimationImpl( const std::vector<Mat>& responses ) = 0;
  virtual void modelUpdateImpl() = 0;

};


class CV_EXPORTS_W Tracker : public virtual Algorithm
{
 public:

  virtual ~Tracker();

  /** @brief Initialize the tracker with a known bounding box that surrounded the target
    @param image The initial frame
    @param boundingBox The initial bounding box

    @return True if initialization went succesfully, false otherwise
     */
  CV_WRAP bool init( InputArray image, const Rect2d& boundingBox );

  /** @brief Update the tracker, find the new most likely bounding box for the target
    @param image The current frame
    @param boundingBox The bounding box that represent the new target location, if true was returned, not
    modified otherwise

    @return True means that target was located and false means that tracker cannot locate target in
    current frame. Note, that latter *does not* imply that tracker has failed, maybe target is indeed
    missing from the frame (say, out of sight)
     */
  CV_WRAP bool update( InputArray image, CV_OUT Rect2d& boundingBox );

  virtual void read( const FileNode& fn )=0;
  virtual void write( FileStorage& fs ) const=0;

 protected:

  virtual bool initImpl( const Mat& image, const Rect2d& boundingBox ) = 0;
  virtual bool updateImpl( const Mat& image, Rect2d& boundingBox ) = 0;

  bool isInit;

  Ptr<TrackerModel> model;
};

Tracker::~Tracker()
{
}

bool Tracker::init( InputArray image, const Rect2d& boundingBox )
{

  if( isInit )
  {
    return false;
  }

  if( image.empty() )
    return false;

  model = Ptr<TrackerModel>();

  bool initTracker = initImpl( image.getMat(), boundingBox );

  //check if the model component is initialized
  if( model == 0 )
  {
    CV_Error( -1, "The model is not initialized" );
    return false;
  }
  if( initTracker )
  {
    isInit = true;
  }
  return initTracker;
}

bool Tracker::update( InputArray image, Rect2d& boundingBox )
{
  if( !isInit )
  {
    return false;
  }
  if( image.empty() )
    return false;
  return updateImpl( image.getMat(), boundingBox );
}

class CV_EXPORTS_W TrackerMedianFlow : public Tracker
{
 public:
  struct CV_EXPORTS Params
  {
    Params(); //!<default constructor
              //!<note that the default values of parameters are recommended for most of use cases
    int pointsInGrid;      //!<square root of number of keypoints used; increase it to trade
                           //!<accurateness for speed
    cv::Size winSize;      //!<window size parameter for Lucas-Kanade optical flow
    int maxLevel;          //!<maximal pyramid level number for Lucas-Kanade optical flow
    TermCriteria termCriteria; //!<termination criteria for Lucas-Kanade optical flow
    cv::Size winSizeNCC;   //!<window size around a point for normalized cross-correlation check
    double maxMedianLengthOfDisplacementDifference; //!<criterion for loosing the tracked object

    void read( const FileNode& /*fn*/ );
    void write( FileStorage& /*fs*/ ) const;
  };

  /** @brief Constructor
    @param parameters Median Flow parameters TrackerMedianFlow::Params
    */
  static Ptr<TrackerMedianFlow> create(const TrackerMedianFlow::Params &parameters);

  CV_WRAP static Ptr<TrackerMedianFlow> create();

  virtual ~TrackerMedianFlow() {}
};

}

namespace
{
using namespace cv;

/*
 *  TrackerMedianFlow
 */
/*
 * TODO:
 * add "non-detected" answer in algo --> test it with 2 rects --> frame-by-frame debug in TLD --> test it!!
 * take all parameters out
 *              asessment framework
 *
 *
 * FIXME:
 * when patch is cut from image to compute NCC, there can be problem with size
 * optimize (allocation<-->reallocation)
 */

class TrackerMedianFlowImpl : public TrackerMedianFlow{
public:
    TrackerMedianFlowImpl(TrackerMedianFlow::Params paramsIn = TrackerMedianFlow::Params()) {params=paramsIn;isInit=false;}
    void read( const FileNode& fn );
    void write( FileStorage& fs ) const;
private:
    bool initImpl( const Mat& image, const Rect2d& boundingBox );
    bool updateImpl( const Mat& image, Rect2d& boundingBox );
    bool medianFlowImpl(Mat oldImage,Mat newImage,Rect2d& oldBox);
    Rect2d vote(const std::vector<Point2f>& oldPoints,const std::vector<Point2f>& newPoints,const Rect2d& oldRect,Point2f& mD);
    float dist(Point2f p1,Point2f p2);
    std::string type2str(int type);
#if 0
    void computeStatistics(std::vector<float>& data,int size=-1);
#endif
    void check_FB(const std::vector<Mat>& oldImagePyr,const std::vector<Mat>& newImagePyr,
                  const std::vector<Point2f>& oldPoints,const std::vector<Point2f>& newPoints,std::vector<bool>& status);
    void check_NCC(const Mat& oldImage,const Mat& newImage,
                   const std::vector<Point2f>& oldPoints,const std::vector<Point2f>& newPoints,std::vector<bool>& status);

    TrackerMedianFlow::Params params;
};

Mat getPatch(Mat image, Size patch_size, Point2f patch_center)
{
    Mat patch;
    Point2i roi_strat_corner(cvRound(patch_center.x - patch_size.width / 2.),
            cvRound(patch_center.y - patch_size.height / 2.));

    Rect2i patch_rect(roi_strat_corner, patch_size);

    if(patch_rect == (patch_rect & Rect2i(0, 0, image.cols, image.rows)))
    {
        patch = image(patch_rect);
    }
    else
    {
        getRectSubPix(image, patch_size,
                      Point2f((float)(patch_rect.x + patch_size.width  / 2.),
                              (float)(patch_rect.y + patch_size.height / 2.)), patch);
    }

    return patch;
}

class TrackerMedianFlowModel : public TrackerModel{
public:
    TrackerMedianFlowModel(TrackerMedianFlow::Params /*params*/){}
    Rect2d getBoundingBox(){return boundingBox_;}
    void setBoudingBox(Rect2d boundingBox){boundingBox_=boundingBox;}
    Mat getImage(){return image_;}
    void setImage(const Mat& image){image.copyTo(image_);}
protected:
    Rect2d boundingBox_;
    Mat image_;
    void modelEstimationImpl( const std::vector<Mat>& /*responses*/ ){}
    void modelUpdateImpl(){}
};

void TrackerMedianFlowImpl::read( const cv::FileNode& fn )
{
    params.read( fn );
}

void TrackerMedianFlowImpl::write( cv::FileStorage& fs ) const
{
    params.write( fs );
}

bool TrackerMedianFlowImpl::initImpl( const Mat& image, const Rect2d& boundingBox ){
    model=Ptr<TrackerMedianFlowModel>(new TrackerMedianFlowModel(params));
    ((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->setImage(image);
    ((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->setBoudingBox(boundingBox);
    return true;
}

bool TrackerMedianFlowImpl::updateImpl( const Mat& image, Rect2d& boundingBox ){
    Mat oldImage=((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->getImage();

    Rect2d oldBox=((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->getBoundingBox();
    if(!medianFlowImpl(oldImage,image,oldBox)){
        return false;
    }
    boundingBox=oldBox;
    ((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->setImage(image);
    ((TrackerMedianFlowModel*)static_cast<TrackerModel*>(model))->setBoudingBox(oldBox);
    return true;
}

template<typename T>
size_t filterPointsInVectors(std::vector<T>& status, std::vector<Point2f>& vec1, std::vector<Point2f>& vec2, T goodValue)
{
    CV_DbgAssert(status.size() == vec1.size() && status.size() == vec2.size());

    size_t first_bad_idx = 0;
    while(first_bad_idx < status.size())
    {
        if(status[first_bad_idx] != goodValue)
            break;
        first_bad_idx++;
    }

    if (first_bad_idx >= status.size())
        return first_bad_idx;

    for(size_t i = first_bad_idx + 1; i < status.size(); i++)
    {
        if (status[i] != goodValue)
            continue;

        status[first_bad_idx] = goodValue;
        vec1[first_bad_idx] = vec1[i];
        vec2[first_bad_idx] = vec2[i];
        first_bad_idx++;
    }
    vec1.erase(vec1.begin() + first_bad_idx, vec1.end());
    vec2.erase(vec2.begin() + first_bad_idx, vec2.end());
    status.erase(status.begin() + first_bad_idx, status.end());

    return first_bad_idx;
}

bool TrackerMedianFlowImpl::medianFlowImpl(Mat oldImage,Mat newImage,Rect2d& oldBox){
    std::vector<Point2f> pointsToTrackOld,pointsToTrackNew;

    Mat oldImage_gray,newImage_gray;
    if (oldImage.channels() != 1)
        cvtColor( oldImage, oldImage_gray, COLOR_BGR2GRAY );
    else
        oldImage.copyTo(oldImage_gray);

    if (newImage.channels() != 1)
        cvtColor( newImage, newImage_gray, COLOR_BGR2GRAY );
    else
        newImage.copyTo(newImage_gray);

    //"open ended" grid
    for(int i=0;i<params.pointsInGrid;i++){
        for(int j=0;j<params.pointsInGrid;j++){
            pointsToTrackOld.push_back(
                        Point2f((float)(oldBox.x+((1.0*oldBox.width)/params.pointsInGrid)*j+.5*oldBox.width/params.pointsInGrid),
                                (float)(oldBox.y+((1.0*oldBox.height)/params.pointsInGrid)*i+.5*oldBox.height/params.pointsInGrid)));
        }
    }

    std::vector<uchar> status(pointsToTrackOld.size());
    std::vector<float> errors(pointsToTrackOld.size());

    std::vector<Mat> oldImagePyr;
    buildOpticalFlowPyramid(oldImage_gray, oldImagePyr, params.winSize, params.maxLevel, false);

    std::vector<Mat> newImagePyr;
    buildOpticalFlowPyramid(newImage_gray, newImagePyr, params.winSize, params.maxLevel, false);

    calcOpticalFlowPyrLK(oldImagePyr,newImagePyr,pointsToTrackOld,pointsToTrackNew,status,errors,
                         params.winSize, params.maxLevel, params.termCriteria, 0);

    CV_Assert(pointsToTrackNew.size() == pointsToTrackOld.size());
    CV_Assert(status.size() == pointsToTrackOld.size());
    //dprintf(("\t%d after LK forward\n",(int)pointsToTrackOld.size()));

    size_t num_good_points_after_optical_flow = filterPointsInVectors(status, pointsToTrackOld, pointsToTrackNew, (uchar)1);

    //dprintf(("\t num_good_points_after_optical_flow = %d\n",num_good_points_after_optical_flow));

    if (num_good_points_after_optical_flow == 0) {
        return false;
    }

    CV_Assert(pointsToTrackOld.size() == num_good_points_after_optical_flow);
    CV_Assert(pointsToTrackNew.size() == num_good_points_after_optical_flow);

    //dprintf(("\t%d after LK forward after removing points with bad status\n",(int)pointsToTrackOld.size()));

    std::vector<bool> filter_status(pointsToTrackOld.size(), true);
    check_FB(oldImagePyr, newImagePyr, pointsToTrackOld, pointsToTrackNew, filter_status);
    check_NCC(oldImage_gray, newImage_gray, pointsToTrackOld, pointsToTrackNew, filter_status);

    // filter
    size_t num_good_points_after_filtering = filterPointsInVectors(filter_status, pointsToTrackOld, pointsToTrackNew, true);

    //dprintf(("\t num_good_points_after_filtering = %d\n",num_good_points_after_filtering));

    if(num_good_points_after_filtering == 0){
        return false;
    }

    CV_Assert(pointsToTrackOld.size() == num_good_points_after_filtering);
    CV_Assert(pointsToTrackNew.size() == num_good_points_after_filtering);

    //dprintf(("\t%d after LK backward\n",(int)pointsToTrackOld.size()));

    std::vector<Point2f> di(pointsToTrackOld.size());
    for(size_t i=0; i<pointsToTrackOld.size(); i++){
        di[i] = pointsToTrackNew[i]-pointsToTrackOld[i];
    }

    Point2f mDisplacement;
    oldBox=vote(pointsToTrackOld,pointsToTrackNew,oldBox,mDisplacement);

    std::vector<float> displacements;
    for(size_t i=0;i<di.size();i++){
        di[i]-=mDisplacement;
        displacements.push_back((float)sqrt(di[i].ddot(di[i])));
    }
    float median_displacements = tracking_internal::getMedianAndDoPartition(displacements);
    //dprintf(("\tmedian of length of difference of displacements = %f\n", median_displacements));
    if(median_displacements > params.maxMedianLengthOfDisplacementDifference){
    //    dprintf(("\tmedian flow tracker returns false due to big median length of difference between displacements\n"));
        return false;
    }

    return true;
}

Rect2d TrackerMedianFlowImpl::vote(const std::vector<Point2f>& oldPoints,const std::vector<Point2f>& newPoints,const Rect2d& oldRect,Point2f& mD){
    Rect2d newRect;
    Point2d newCenter(oldRect.x+oldRect.width/2.0,oldRect.y+oldRect.height/2.0);
    const size_t n=oldPoints.size();

    if (n==1) {
        newRect.x=oldRect.x+newPoints[0].x-oldPoints[0].x;
        newRect.y=oldRect.y+newPoints[0].y-oldPoints[0].y;
        newRect.width=oldRect.width;
        newRect.height=oldRect.height;
        mD.x = newPoints[0].x-oldPoints[0].x;
        mD.y = newPoints[0].y-oldPoints[0].y;
        return newRect;
    }

    float xshift=0,yshift=0;
    std::vector<float> buf_for_location(n, 0.);
    for(size_t i=0;i<n;i++){  buf_for_location[i]=newPoints[i].x-oldPoints[i].x;  }
    xshift=tracking_internal::getMedianAndDoPartition(buf_for_location);
    newCenter.x+=xshift;
    for(size_t i=0;i<n;i++){  buf_for_location[i]=newPoints[i].y-oldPoints[i].y;  }
    yshift=tracking_internal::getMedianAndDoPartition(buf_for_location);
    newCenter.y+=yshift;
    mD=Point2f((float)xshift,(float)yshift);

    std::vector<double> buf_for_scale(n*(n-1)/2, 0.0);
    for(size_t i=0,ctr=0;i<n;i++){
        for(size_t j=0;j<i;j++){
            double nd=norm(newPoints[i] - newPoints[j]);
            double od=norm(oldPoints[i] - oldPoints[j]);
            buf_for_scale[ctr]=(od==0.0)?0.0:(nd/od);
            ctr++;
        }
    }

    double scale=tracking_internal::getMedianAndDoPartition(buf_for_scale);
    //dprintf(("xshift, yshift, scale = %f %f %f\n",xshift,yshift,scale));
    newRect.x=newCenter.x-scale*oldRect.width/2.0;
    newRect.y=newCenter.y-scale*oldRect.height/2.0;
    newRect.width=scale*oldRect.width;
    newRect.height=scale*oldRect.height;
    //dprintf(("rect old [%f %f %f %f]\n",oldRect.x,oldRect.y,oldRect.width,oldRect.height));
    //dprintf(("rect [%f %f %f %f]\n",newRect.x,newRect.y,newRect.width,newRect.height));

    return newRect;
}
#if 0
void TrackerMedianFlowImpl::computeStatistics(std::vector<float>& data,int size){
    int binnum=10;
    if(size==-1){
        size=(int)data.size();
    }
    float mini=*std::min_element(data.begin(),data.begin()+size),maxi=*std::max_element(data.begin(),data.begin()+size);
    std::vector<int> bins(binnum,(int)0);
    for(int i=0;i<size;i++){
        bins[std::min((int)(binnum*(data[i]-mini)/(maxi-mini)),binnum-1)]++;
    }
    for(int i=0;i<binnum;i++){
        //dprintf(("[%4f,%4f] -- %4d\n",mini+(maxi-mini)/binnum*i,mini+(maxi-mini)/binnum*(i+1),bins[i]));
    }
}
#endif
void TrackerMedianFlowImpl::check_FB(const std::vector<Mat>& oldImagePyr, const std::vector<Mat>& newImagePyr,
                                     const std::vector<Point2f>& oldPoints, const std::vector<Point2f>& newPoints, std::vector<bool>& status){

    if(status.empty()) {
        status=std::vector<bool>(oldPoints.size(),true);
    }

    std::vector<uchar> LKstatus(oldPoints.size());
    std::vector<float> errors(oldPoints.size());
    std::vector<float> FBerror(oldPoints.size());
    std::vector<Point2f> pointsToTrackReprojection;
    calcOpticalFlowPyrLK(newImagePyr, oldImagePyr,newPoints,pointsToTrackReprojection,LKstatus,errors,
                         params.winSize, params.maxLevel, params.termCriteria, 0);

    for(size_t i=0;i<oldPoints.size();i++){
        FBerror[i]=(float)norm(oldPoints[i]-pointsToTrackReprojection[i]);
    }
    float FBerrorMedian=tracking_internal::getMedian(FBerror);
    //dprintf(("point median=%f\n",FBerrorMedian));
    //dprintf(("FBerrorMedian=%f\n",FBerrorMedian));
    for(size_t i=0;i<oldPoints.size();i++){
        status[i]=status[i] && (FBerror[i] <= FBerrorMedian);
    }
}
void TrackerMedianFlowImpl::check_NCC(const Mat& oldImage,const Mat& newImage,
                                      const std::vector<Point2f>& oldPoints,const std::vector<Point2f>& newPoints,std::vector<bool>& status){

    std::vector<float> NCC(oldPoints.size(),0.0);
    Mat p1,p2;

    for (size_t i = 0; i < oldPoints.size(); i++) {
        p1 = getPatch(oldImage, params.winSizeNCC, oldPoints[i]);
        p2 = getPatch(newImage, params.winSizeNCC, newPoints[i]);

        NCC[i] = (float)tracking_internal::computeNCC(p1, p2);
    }
    float median = tracking_internal::getMedian(NCC);
    for(size_t i = 0; i < oldPoints.size(); i++) {
        status[i] = status[i] && (NCC[i] >= median);
    }
}

} /* anonymous namespace */

namespace cv
{
/*
 * Parameters
 */
TrackerMedianFlow::Params::Params() {
    pointsInGrid=10;
    winSize = Size(3,3);
    maxLevel = 5;
    termCriteria = TermCriteria(TermCriteria::COUNT|TermCriteria::EPS,20,0.3);
    winSizeNCC = Size(30,30);
    maxMedianLengthOfDisplacementDifference = 10;
}

void TrackerMedianFlow::Params::read( const cv::FileNode& fn ){
    *this = TrackerMedianFlow::Params();

    if (!fn["winSize"].empty())
        fn["winSize"] >> winSize;

    if(!fn["winSizeNCC"].empty())
        fn["winSizeNCC"] >> winSizeNCC;

    if(!fn["pointsInGrid"].empty())
        fn["pointsInGrid"] >> pointsInGrid;

    if(!fn["maxLevel"].empty())
        fn["maxLevel"] >> maxLevel;

    if(!fn["maxMedianLengthOfDisplacementDifference"].empty())
        fn["maxMedianLengthOfDisplacementDifference"] >> maxMedianLengthOfDisplacementDifference;

    if(!fn["termCriteria_maxCount"].empty())
        fn["termCriteria_maxCount"] >> termCriteria.maxCount;

    if(!fn["termCriteria_epsilon"].empty())
        fn["termCriteria_epsilon"] >> termCriteria.epsilon;
}

void TrackerMedianFlow::Params::write( cv::FileStorage& fs ) const{
    fs << "pointsInGrid" << pointsInGrid;
    fs << "winSize" << winSize;
    fs << "maxLevel" << maxLevel;
    fs << "termCriteria_maxCount" << termCriteria.maxCount;
    fs << "termCriteria_epsilon" << termCriteria.epsilon;
    fs << "winSizeNCC" << winSizeNCC;
    fs << "maxMedianLengthOfDisplacementDifference" << maxMedianLengthOfDisplacementDifference;
}

Ptr<TrackerMedianFlow> TrackerMedianFlow::create(const TrackerMedianFlow::Params &parameters){
    return Ptr<TrackerMedianFlowImpl>(new TrackerMedianFlowImpl(parameters));
}
Ptr<TrackerMedianFlow> TrackerMedianFlow::create(){
    return Ptr<TrackerMedianFlowImpl>(new TrackerMedianFlowImpl());
}

} /* namespace cv */
