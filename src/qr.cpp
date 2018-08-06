using namespace std;
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "qr.h"
#include <vector>
 extern"C"  
{  
    #include<cblas.h>     
} 

int sunt = 0;
float enengy(cv::Mat a ,cv::Mat b)
{
	//clock_t start = clock();
	float* aData = a.ptr<float>(0,0);
	float* bData = b.ptr<float>(0,0);

	float out = 0.0f;
	for (int i=0 ; i<b.cols;i++)
		out += *aData++ * *bData++;
	//sunt+=int(clock() - start );
	//cout << sunt << endl;
	return out;
	//cv::Mat c = a * b.reshape(1,b.cols);


	//return c.at<float>(0,0);
}



vector<cv::Mat> ypQR_Schmidt(const cv::Mat AA)
{


/*

     blas版
	int width = AA.cols;
	int height  = AA.rows;
	float AA_blas[height*width];
	for (int ii = 0; ii< width; ii++)
		for (int jj = 0; jj < height; jj++)
			AA_blas[ii*height + jj] = AA.at<float>(jj,ii);


	for (int i = 0; i < AA.cols ; i++)
	{
		float* temp_A= AA_blas + i * height;
		for (int j = 0; j < i ; j++)
		{
			float* temp_B= AA_blas + j * height;
			cblas_saxpy(height ,- cblas_sdot(height ,temp_B ,1 ,temp_A ,1)/cblas_sdot(height ,temp_B ,1 ,temp_B ,1), temp_B , 1, temp_A ,1);
		}
		cblas_sscal(height, 1.0f/cblas_snrm2(height,temp_A,1), temp_A, 1);
	}

	
   gpu版

	cv::Mat res(AA.rows,AA.cols,AA.type(),cv::Scalar(0));

	for (int ii = 0; ii< width; ii++)
		for (int jj = 0; jj < height; jj++)
			res.at<float>(jj,ii) = AA_blas[ii*height + jj] ;

     cv::cuda::Stream stream;
	cv::cuda::GpuMat G_AA ;
	cv::cuda::GpuMat  nn;
	
	G_AA.upload(AA);

	for (int i =0 ; i< AA.cols ; i++)
	{
		for(int j = 0; j < i ; j++)
		{
		cv::cuda::addWeighted(G_AA.col(i), 1.0f, G_AA.col(j) ,  g_enengy(G_AA.col(i),G_AA.col(j)), 0.0, G_AA.col(i), -1, stream);
		}
		cv::cuda::normalize(G_AA.col(i),G_AA.col(i),1 ,0 , cv::NORM_L2, -1 , cv::cuda::GpuMat(), stream);

	}
        G_AA.download(res);
*/ 
	cv::Mat Q;
	vector<cv::Mat> res;
	AA.copyTo(Q);
	cv::transpose(Q,Q);
	cv::Mat R(Q.rows,Q.rows,CV_32FC1,cv::Scalar(0));
	for (int i =0 ; i< Q.rows ; i++)
	{
		cv::Mat temp = Q.row(i);

		for(int j = 0; j < i ; j++)
		{
			float r = enengy(Q.row(j), Q.row(i)) /enengy(Q.row(j) , Q.row(j) );
			R.at<float>(j,i) = r ;
			temp = temp - r *  Q.row(j);
		}
		float c = sqrt(enengy(temp , temp)) ;
		R.at<float>(i ,i) = c ;
		temp = temp /  c;
		
		//cv::normalize(temp , temp , 1,0, cv::NORM_L2);
	        
 	}
	res.push_back(Q);
	res.push_back(R);


    return res;
}




