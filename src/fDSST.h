#ifndef FDSST
#define FDSST
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include "params.hpp"
#include "feature_type.h"
#include "fftTool.h"
#include "FHOG.hpp"
#include "fhog_f.hpp"
#include "qr.h"
#include <math.h>
#include <cmath>


class fDSST
{
public:
	fDSST(){};

	virtual    ~fDSST(){};
	cv::Mat  scaleSizeFactors;
	cv::Mat  interpScaleFactors;
	cv::Mat  yf;
	cv::Mat  s_num;
	cv::Mat  cos_window;
	cv::Mat  sf_den;
	cv::Mat  basis;
	cv::Mat  sf_num;
	int  nScales;
	float  scale_step;
	float  scale_sigma;
	cv::Size  init_target_sz;	
	cv::Size  scale_model_sz;
	eco_params  params;
	void  init_scale_filter(eco_params& pparams);
	void  scale_filter_update(cv::Mat im, cv::Point2f pos, cv::Size2f base_target_sz, float currentScaleFactor);
	float  scale_filter_track(cv::Mat im, cv::Point2f pos, cv::Size2f base_target_sz, float currentScaleFactor);
	cv::Mat  resizeDFT(cv::Mat inputdft, int desiredLen);
	cv::Mat  extract_scale_sample(cv::Mat im, cv::Point2f pos, cv::Size2f base_target_sz, cv::Mat scaleFactors);
};




#endif
