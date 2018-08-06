#ifndef FEATURE_EXTRACTOR
#define FEATURE_EXTRACTOR

#include <iostream>
#include <string>
#include <math.h>
#include <vector>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <numeric>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include "params.hpp"
#include "feature_type.h"
#include "fftTool.h"
#include "recttools.hpp"
#include "FHOG.hpp"
#include "fhog_f.hpp"

using namespace FFTTools;
using namespace std;


class feature_extractor
{
public:
	feature_extractor(){}

	virtual    ~feature_extractor(){};

	void  init(const eco_params& params);

	ECO_FEATS  extractor(cv::Mat image, cv::Point2f pos, vector<float> scales);

	cv::Mat    sample_patch(const cv::Mat& im, const cv::Point2f& pos, cv::Size2f sample_sz, cv::Size2f output_sz);

	vector<cv::Mat>   get_hog(vector<cv::Mat> im);

	vector<cv::Mat>   get_cn(vector<cv::Mat> im);

	vector<cv::Mat>   hog_feature_normalization(vector<cv::Mat>& feature);

	vector<cv::Mat>   cn_feature_normalization(vector<cv::Mat>& cn_feat_maps);
	

	cv::Mat			  sample_pool(const cv::Mat& im, int smaple_factor, int stride);
	

	inline vector<cv::Mat>  get_hog_feats()const { return hog_feat_maps; }

private: 

	hog_feature							hog_features;

	cn_feature							cn_features;

	int 								cn_cell_size;

	bool 							useCnFeature = true;

	vector<cv::Mat>                     hog_feat_maps;

	vector<cv::Mat>                     cn_feat_maps;	


};



#endif
 
