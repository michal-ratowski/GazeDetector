#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace std;
using namespace cv;

int H_MIN = 0;
int H_MAX = 256;
int S_MIN = 0;
int S_MAX = 256;
int V_MIN = 0;
int V_MAX = 256;
int CENTER_VERTICAL = 0;
int CENTER_HORIZONTAL = 0;
int DELTA_VERTICAL = 0;
int DELTA_HORIZONTAL = 0;


const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;

void on_trackbar( int, void* )
{
}

void createTrackbars(){

    namedWindow("HSV trackbars",0);

    char TrackbarName[50];

    sprintf( TrackbarName, "H_MIN", H_MIN);
    sprintf( TrackbarName, "H_MAX", H_MAX);
    sprintf( TrackbarName, "S_MIN", S_MIN);
    sprintf( TrackbarName, "S_MAX", S_MAX);
    sprintf( TrackbarName, "V_MIN", V_MIN);
    sprintf( TrackbarName, "V_MAX", V_MAX);
    sprintf( TrackbarName, "CENTER_V", CENTER_VERTICAL);
    sprintf( TrackbarName, "CENTER_H", CENTER_HORIZONTAL);

    createTrackbar( "H_MIN", "HSV trackbars", &H_MIN, H_MAX, on_trackbar );
    createTrackbar( "H_MAX", "HSV trackbars", &H_MAX, H_MAX, on_trackbar );
    createTrackbar( "S_MIN", "HSV trackbars", &S_MIN, S_MAX, on_trackbar );
    createTrackbar( "S_MAX", "HSV trackbars", &S_MAX, S_MAX, on_trackbar );
    createTrackbar( "V_MIN", "HSV trackbars", &V_MIN, V_MAX, on_trackbar );
    createTrackbar( "V_MAX", "HSV trackbars", &V_MAX, V_MAX, on_trackbar );
    createTrackbar( "CENTER_V", "HSV trackbars", &CENTER_VERTICAL, 100, on_trackbar );
    createTrackbar( "CENTER_H", "HSV trackbars", &CENTER_HORIZONTAL, 100, on_trackbar );



}

void drawCross(int x, int y,Mat &frame){

    if(y-5>0) line(frame,Point(x,y),Point(x,y-5),Scalar(0,255,0),1);
    else line(frame,Point(x,y),Point(x,0),Scalar(0,255,0),1);

    if(y+5<FRAME_HEIGHT) line(frame,Point(x,y),Point(x,y+5),Scalar(0,255,0),1);
    else line(frame,Point(x,y),Point(x,FRAME_HEIGHT),Scalar(0,255,0),1);

    if(x-5>0) line(frame,Point(x,y),Point(x-5,y),Scalar(0,255,0),1);
    else line(frame,Point(x,y),Point(0,y),Scalar(0,255,0),1);

    if(x+5<FRAME_WIDTH) line(frame,Point(x,y),Point(x+5,y),Scalar(0,255,0),1);
    else line(frame,Point(x,y),Point(FRAME_WIDTH,y),Scalar(0,255,0),1);

}


int main(int argc, char* argv[]) {

    Mat cameraImage,leftEyeThreshold,rightEyeThreshold,pupilThreshold,pupilrightEyeThreshold,cornersImage;

    cv::SimpleBlobDetector::Params params;
    params.minDistBetweenBlobs = 50.0f;
    params.filterByInertia = false;
    params.filterByConvexity = false;
    params.filterByColor = false;
    params.filterByCircularity = false;
    params.filterByArea = true;
    params.minArea = 50.0f;
    params.maxArea = 500.0f;
    cv::SimpleBlobDetector blob_detector(params);

    KalmanFilter KF(4, 2, 0);
    KF.transitionMatrix = *(Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);
    Mat_<float> measurement(2,1); measurement.setTo(Scalar(0));

    KF.statePre.at<float>(0) = 55;
    KF.statePre.at<float>(1) = 26;
    KF.statePre.at<float>(2) = 0;
    KF.statePre.at<float>(3) = 0;
    setIdentity(KF.measurementMatrix);
    setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
    setIdentity(KF.measurementNoiseCov, Scalar::all(0.1));
    setIdentity(KF.errorCovPost, Scalar::all(.1));

    CascadeClassifier* faceCascade=new CascadeClassifier("/home/tomekfasola/Projekty/opencv/haarcascade_frontalface_alt2.xml");
    CascadeClassifier* eyeCascade=new CascadeClassifier("/home/tomekfasola/Projekty/opencv/haarcascade_eye.xml");

    createTrackbars();
    VideoCapture capture;
    capture.open(0);

    capture.set(CV_CAP_PROP_FRAME_WIDTH,FRAME_WIDTH);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT,FRAME_HEIGHT);

    while(1){

        capture.read(cameraImage);

        vector< Rect_<int> > faces;
        vector< Rect_<int> > eyes;

        faceCascade->detectMultiScale(cameraImage, faces,1.3, 5);

        if(faces.size()>0) {

            if(faces[0].width>100 && faces[0].height>100){

               Rect face_i = faces[0];
               rectangle(cameraImage, face_i, CV_RGB(0, 255,0), 2);

               Mat leftROI = cameraImage(Rect(face_i.x,face_i.y,face_i.width/2,face_i.height/2));
               Mat rightROI= cameraImage(Rect(face_i.x+face_i.width/2,face_i.y,face_i.width/2,face_i.height/2));

               rectangle(cameraImage, Rect(face_i.x,face_i.y+face_i.height/4,face_i.width/2,face_i.height/4), CV_RGB(255, 255,255), 1);
               rectangle(cameraImage, Rect(face_i.x+face_i.width/2,face_i.y+face_i.height/4,face_i.width/2,face_i.height/4), CV_RGB(255, 255,255), 1);

               eyeCascade->detectMultiScale(leftROI, eyes);
               Rect eye_i;

               if(eyes.size()>0)
               {
                    eye_i=eyes[0];

                    rectangle(leftROI,Rect(eye_i.x,eye_i.y,eye_i.width,eye_i.height),CV_RGB(255, 0,0), 1);

                    Mat hsv = leftROI(Rect(eye_i.x,eye_i.y+eye_i.height/4,eye_i.width,eye_i.height*3/4));

                    cornersImage = leftROI(Rect(eye_i.x,eye_i.y+eye_i.height/4,eye_i.width,eye_i.height*3/4));
                    cvtColor(cornersImage,cornersImage,COLOR_BGR2GRAY);
                    equalizeHist(cornersImage,cornersImage);
                    erode(cornersImage,cornersImage, Mat(), Point(-1, -1), 2, 1, 1);

                    inRange(hsv,Scalar(H_MIN,S_MIN,V_MIN),Scalar(H_MAX,S_MAX,V_MAX),leftEyeThreshold);

                    dilate(leftEyeThreshold,leftEyeThreshold, Mat(), Point(-1, -1), 2, 1, 1);

                    pupilThreshold = leftEyeThreshold.clone();

                    vector<cv::KeyPoint> keypoints;
                    blob_detector.detect(leftEyeThreshold, keypoints);

                    if(!keypoints.size()==0)
                    {
                        /*
                        vector< vector<Point> > contours;
                        vector<Vec4i> hierarchy;

                        findContours(temp,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE );
                        */

                        //////////////////////////////// KALMAN
                        Mat prediction = KF.predict();
                        Point predictPt(prediction.at<float>(0),prediction.at<float>(1));

                        //cout<<(keypoints[0].pt.x/eye_i.width)<<" , "<<keypoints[0].pt.y/eye_i.height<<endl;

                        measurement(0) = 100*keypoints[0].pt.x/eye_i.width;
                        measurement(1) = 100*keypoints[0].pt.y/eye_i.height;

                        Point measPt(measurement(0),measurement(1));
                        Mat estimated = KF.correct(measurement);
                        Point statePt(estimated.at<float>(0),estimated.at<float>(1));

                        drawCross(face_i.x+eye_i.x+keypoints[0].pt.x,face_i.y+eye_i.y+eye_i.height/4+keypoints[0].pt.y,cameraImage);

                        /////////////////////////////////////////////////////////

                        std::ostringstream position;

                        if(statePt.x<CENTER_HORIZONTAL-DELTA_HORIZONTAL){
                            if(statePt.y<CENTER_VERTICAL-DELTA_VERTICAL){
                                position<<"TOP-RIGHT";
                            }
                            else if(statePt.y>CENTER_VERTICAL+DELTA_VERTICAL){
                                position<<"BOTTOM-RIGHT";
                            }
                            else{
                                position<< "RIGHT";
                            }

                        }
                        else if(statePt.x>CENTER_HORIZONTAL+DELTA_HORIZONTAL){
                            if(statePt.y<CENTER_VERTICAL-DELTA_VERTICAL){
                                position<< "TOP-LEFT";
                            }
                            else if(statePt.y>CENTER_VERTICAL+DELTA_VERTICAL){
                                position<< "BOTTOM-LEFT";
                            }
                            else{
                                position<< "LEFT";
                            }

                        }
                        else {
                            if(statePt.y<CENTER_VERTICAL-DELTA_VERTICAL){
                                position<< "TOP";
                            }
                            else if(statePt.y>CENTER_VERTICAL+DELTA_VERTICAL){
                                position<< "BOTTOM";
                            }
                            else{
                                position<< "CENTER";
                            }

                        }

                        std::ostringstream oss3;
                        oss3 <<std::setprecision(4) <<"Current: ("<<100*keypoints[0].pt.x/eye_i.width<<","<<100*keypoints[0].pt.y/eye_i.height<<")";
                        std::string var3 = oss3.str();

                        putText(cameraImage,var3,Point(10,450),1,1,Scalar(255,255,255),1);

                        std::ostringstream oss2;
                        oss2 <<std::setprecision(4) <<"Kalman: ("<<statePt.x<<","<<statePt.y<<") Position: "<<position.str();
                        std::string var2 = oss2.str();

                        putText(cameraImage,var2,Point(10,470),1,1,Scalar(255,255,255),1);

                    }
               }

               eyes.clear();
               eyeCascade->detectMultiScale(rightROI, eyes);

               if(eyes.size()>0)
               {
                    Rect eye_r = eyes[0];

                    rectangle(rightROI,Rect(eye_r.x,eye_r.y,eye_r.width,eye_r.height),CV_RGB(255, 0,0), 1);
                    Mat hsv = rightROI(Rect(eye_r.x,eye_r.y+eye_r.height/4,eye_r.width,eye_r.height*3/4));

                    inRange(hsv,Scalar(H_MIN,S_MIN,V_MIN),Scalar(H_MAX,S_MAX,V_MAX),rightEyeThreshold);

                    dilate(rightEyeThreshold,rightEyeThreshold, Mat(), Point(-1, -1), 2, 1, 1);
                    pupilrightEyeThreshold = rightEyeThreshold.clone();

                    vector<cv::KeyPoint> keypoints;
                    blob_detector.detect(rightEyeThreshold, keypoints);

                    if(!keypoints.size()==0)
                    {
                        drawCross(face_i.x+face_i.width/2+eye_r.x+keypoints[0].pt.x,face_i.y+eye_r.y+eye_i.height/4+keypoints[0].pt.y,cameraImage);
                        //putText(cameraImage,var2,Point(250,280),1,1,Scalar(0,0,0),1);
                    }

               }
            }
        }

        imshow("Eye",cornersImage);

        imshow("Gaze tracking app",cameraImage);

        if(!pupilThreshold.empty())
        imshow("Left pupil threshold",pupilThreshold);

        waitKey(30);
    }
    return 0;
}
