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


std::vector<std::string> split(const  std::string& s, const std::string& delim)
{
    std::vector<std::string> elems;
    size_t pos = 0;
    size_t len = s.length();
    size_t delim_len = delim.length();
    if (delim_len == 0) return elems;
    while (pos < len)
    {
        int find_pos = s.find(delim, pos);
        if (find_pos < 0)
        {
            elems.push_back(s.substr(pos, len - pos));
            break;
        }
        elems.push_back(s.substr(pos, find_pos - pos));
        pos = find_pos + delim_len;
    }
    return elems;
}


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
    //cv::Rect box(962,202,128,393);
    
    const string proto("VGG/imagenet-vgg-m-2048.prototxt");
    const string model("VGG/VGG_CNN_M_2048.caffemodel");
    const string mean_file("VGG/VGG_mean.binaryproto");
    const std::string mean_yml("VGG/mean.yml");
    cv::Mat frame;

    cv::Rect result;
    cv::namedWindow(WIN_NAME);
    cv::VideoCapture capture;
    capture.open("/home/miaopass/doubleVideo/d16.mp4");



 

   /*

    ifstream infile; 
    infile.open("/home/miaopass/a.txt");
    string s;
    string delim = "\t";
    vector<string> test ;
    int j = 0;
    while(getline(infile,s)){
	test = split(s,delim);
	for(int c = 0;c < 3;c++){
	    for(int i = 0;i < 1280;i++){
		frame.at<cv::Vec3b>(j,i)[c] = atoi(test[c*1280+i].c_str()); 
	    }
	}
	j++;
    }


    //vector<cv::Mat> sp ;
    //cv::split(frame,sp) ;
    //cout << sp[2].row(0) << endl;


cvNamedWindow(WIN_NAME.c_str(), CV_WINDOW_AUTOSIZE);
	cvSetMouseCallback(WIN_NAME.c_str(), mouseHandler, NULL);

    capture >> frame;
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
*/
    //cv::rectangle(frame,box,cv::Scalar(255,0,0),1,1,0);
    //cv::imshow(WIN_NAME,frame);
    //cv::waitKey(0);
    ECO Eco(proto, model, mean_file,mean_yml);  
    //vector<cv::Mat> sp2 ;
    //sp2.push_back(sp[2]); 
    //sp2.push_back(sp[1]); 
    //sp2.push_back(sp[0]); 
    //cv::Mat frame2;
    //cv::merge(sp2,frame) ;
    capture >> frame;

    frame = cv::imread("/home/miaopass/Downloads/ECO-master/d/d16/img/0001.jpg");
    cv::Rect box(500,	70	,105	,540);
    Eco.init(frame, box);
    //cv::imshow("frame" , frame);
    //cv::waitKey(0);
    int a =0;
    clock_t start, end;
    start = clock();
    while (a<2000)
    {   	
        //cout << a <<"a"<<endl;
        a++;
        capture >> frame;
        Eco.process_frame(frame);  
         
    }
    end = clock();
    cout << "FPS:"  << a/((double)(end - start) / CLOCKS_PER_SEC) << endl;
}




