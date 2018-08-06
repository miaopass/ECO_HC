#ifndef ECO_H
#define ECO_H
#include <iostream>
#include <string>
#include <math.h>
#include <ctime>

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>

#include "params.hpp"
#include "feature_type.h"
#include "interpolator.h"
#include "reg_filter.h"
#include "feature_extractor.h"
#include "feature_operator.h"
#include "eco_sample_update.h"
#include "optimize_scores.h"
#include "training.h"
#include "fDSST.h"
#endif
using namespace std;
using namespace FFTTools;
using namespace eco_sample_update;
class ECO
{
public:
    virtual ~ECO(){}

    ECO();

    void init(cv::Mat& im, const cv::Rect& rect);

    void process_frame(const cv::Mat& frame);


    void init_features(); 

    void yf_gaussion(); 

    void cos_wind(); 
    
    ECO_FEATS do_windows_x(const ECO_FEATS& xl, vector<cv::Mat>& cos_win);

    ECO_FEATS interpolate_dft(const ECO_FEATS& xlf, vector<cv::Mat>& interp1_fs, vector<cv::Mat>& interp2_fs);

    ECO_FEATS compact_fourier_coeff(const ECO_FEATS& xf);

    vector<cv::Mat> init_projection_matrix(const ECO_FEATS& init_sample, const vector<int>& compressed_dim, const vector<int>& feature_dim);

    vector<cv::Mat> project_mat_energy(vector<cv::Mat> proj, vector<cv::Mat> yf);

    ECO_FEATS full_fourier_coeff(ECO_FEATS xf);

    ECO_FEATS shift_sample(ECO_FEATS& xf, cv::Point2f shift, std::vector<cv::Mat> kx, std::vector<cv::Mat> ky);
private:
    
    bool useDeepFeature, is_color_image;


    cv::Mat deep_mean_mat,yml_mean;

    size_t output_sz, k1, frameID, frames_since_last_train; 

    cv::Point2f pos;

    eco_params params;     

    cv::Size target_sz, init_target_sz, img_sample_sz, img_support_sz;

    cv::Size2f base_target_sz; 

    float currentScaleFactor; 

    hog_feature hog_features;

    cn_feature cn_features;

    vector<cv::Size> feature_sz, filter_sz;

    vector<int> feature_dim, compressed_dim;

    vector<cv::Mat> ky, kx, yf, cos_window; 

    vector<cv::Mat> interp1_fs, interp2_fs;

    vector<cv::Mat> reg_filter, projection_matrix; 

    vector<float> reg_energy, scaleFactors;

    feature_extractor feat_extrator;

    fDSST fdsst;

    sample_update SampleUpdate;

    ECO_FEATS sample_energy;

    ECO_FEATS  hf_full;
 
    eco_train eco_trainer;
   
};
    

    
