#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ctime>
//#include "kcftracker.hpp"
#include <dirent.h>
#include "ECO.h"
using namespace std;
//using namespace cv;

static string WIN_NAME = "ECO-Tracker";
bool gotBB = false;
bool drawing_box = false;
cv::Rect box;
void mouseHandler(int event, int x, int y, int flags, void *param){
	switch (event){
	case CV_EVENT_MOUSEMOVE:
		if (drawing_box){
			box.width = x - box.x;
			box.height = y - box.y;
		}
		break;
	case CV_EVENT_LBUTTONDOWN:
		drawing_box = true;
		box = cv::Rect(x, y, 0, 0);
		break;
	case CV_EVENT_LBUTTONUP:
		drawing_box = false;
		if (box.width < 0){
			box.x += box.width;
			box.width *= -1;
		}
		if (box.height < 0){
			box.y += box.height;
			box.height *= -1;
		}
		gotBB = true;
		break;
	}
}

void drawBox(cv::Mat& image, cv::Rect box, cv::Scalar color, int thick){
	rectangle(image, cvPoint(box.x, box.y), cvPoint(box.x + box.width, box.y + box.height), color, thick);
}




int main(){

    cv::Mat frame;
    cv::Rect result;
    cv::namedWindow(WIN_NAME);
    //cv::VideoCapture capture;
    //capture.open("your video");


    cvNamedWindow(WIN_NAME.c_str(), CV_WINDOW_AUTOSIZE);
    cvSetMouseCallback(WIN_NAME.c_str(), mouseHandler, NULL);
    //capture >> frame;


    string path = "data/Crossing";
    ifstream frame_name;
    frame_name.open(path + "/output.txt");
    string s;
    getline(frame_name,s);
    
    frame = cv::imread(path + "/img/" + s +".jpg");
    cv::Mat temp;
    frame.copyTo(temp);

    while (!gotBB)
    {
        drawBox(frame, box, cv::Scalar(0, 0, 255), 1);
        cv::imshow(WIN_NAME, frame);
        temp.copyTo(frame);
        if (cvWaitKey(20) == 27)
            return 1;
    }

    cvSetMouseCallback(WIN_NAME.c_str(), NULL, NULL);
    ECO Eco;  
    Eco.init(frame, box);
    //cv::imshow("frame" , frame);
    //cv::waitKey(0);
    int a =0;
    clock_t start, end;
    start = clock();
    while (getline(frame_name,s))
    {   	

        a++;
        //capture >> frame;
	frame = cv::imread(path + "/img/" + s +".jpg");
        Eco.process_frame(frame);  
         
    }
    end = clock();
    cout << "FPS:"  << a/((double)(end - start) / CLOCKS_PER_SEC) << endl;
}




