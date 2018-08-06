#include "ECO.h"


using namespace std;


#define ECO_TRAIN



ECO::ECO()
{
 
	hog_features.fparams.cell_size = 6;
	hog_features.fparams.compressed_dim = 10;
	hog_features.fparams.nOrients = 9;
	hog_features.fparams.nDim = 31;
	hog_features.fparams.penalty = 0;

	cn_features.fparams.cell_size = 4;
	cn_features.fparams.compressed_dim = 3;
	cn_features.fparams.nOrients = 9;
	cn_features.fparams.nDim = 10;
	cn_features.fparams.penalty = 0;	
	cn_features.fparams.use_cn = true;
	// Image sample parameters
	params.search_area_scale = 4.0f;
	params.min_image_sample_size = pow(150,2);
	params.max_image_sample_size = pow(200,2);


	// Detection parameters
	params.refinement_iterations = 1;
	params.newton_iterations = 5;
	params.clamp_position = false;   


	// Learning parameters
	params.output_sigma_factor = 0.0625f;		
	params.learning_rate = 0.009f;	 	 	
	params.nSamples = 30;
	params.sample_replace_strategy = "lowest_prior";
	params.lt_size = 0;
	params.train_gap = 5;
	params.skip_after_frame = 10;
	params.use_detection_sample = true; 

	//Factorized convolution parameters
	params.projection_reg = 1e-7;

	//Conjugate Gradient parameters
	params.CG_iter = 5;
	params.init_CG_iter = 10*15;
	params.init_GN_iter = 10; 
	params.CG_use_FR = false;
	params.CG_standard_alpha = true;
	params.CG_forgetting_rate = 50;
	params.precond_data_param = 0.75f;
	params.precond_reg_param = 0.25f;
	params.precond_proj_param = 40;


	// Regularization window parameters
	params.use_reg_window = true;
	params.reg_window_min = 0.0001f;
	params.reg_window_edge = 0.01f;
	params.reg_window_power = 2;
	params.reg_sparsity_threshold = 0.05f;


	//Interpolation parameters
	params.interpolation_method = "bicubic";
	params.interpolation_bicubic_a = -0.75f;
	params.interpolation_centering = true;
	params.interpolation_windowing = false; 

	// Only used if: params.use_scale_filter = false
	params.number_of_scales = 1;
	params.scale_step = 1.02f; 



	// Scale filter parameters
	params.use_scale_filter = true;
	params.scale_sigma_factor = 0.0625f; 
	params.scale_learning_rate = 0.025f;
	params.number_of_scales_filter = 17;
	params.number_of_interp_scales = 33;
	params.scale_model_factor = 1.0f;
	params.scale_step_filter = 1.02f;
	params.scale_model_max_area = 32*16;
	params.scale_feature = "HOG4";
	params.s_num_compressed_dim = "MAX";
	params.lambda = 0.01f;
	params.do_poly_interp = true;

	params.hog_feat = hog_features; 
	params.cn_feat = cn_features; 
	params.visualization = true;    

}
void ECO::init(cv::Mat& im, const cv::Rect& rect){

    bool debug = false;
    pos.x = rect.x + float(rect.width - 1) / 2;
    pos.y = rect.y + float(rect.height - 1) / 2;
 
    target_sz = rect.size();

    params.init_sz = target_sz; 
    
    int search_area = rect.area() * pow(params.search_area_scale, 2);
    if (search_area > params.max_image_sample_size){
        currentScaleFactor = sqrt((float)search_area / params.max_image_sample_size);
    }
    else if (search_area < params.min_image_sample_size)
        currentScaleFactor = sqrt((float)search_area / params.min_image_sample_size);
    else
        currentScaleFactor = 1.0;

    base_target_sz = cv::Size2f(target_sz.width / currentScaleFactor, target_sz.height / currentScaleFactor);
  
    int max_width = sqrt(params.max_image_sample_size);
    int min_width = sqrt(params.min_image_sample_size);
    if (currentScaleFactor > 1)
        img_sample_sz = cv::Size(max_width, max_width);
    else
        img_sample_sz = cv::Size(min_width, min_width);
    
    init_features();
    img_support_sz = hog_features.img_input_sz;


    if(cn_features.fparams.use_cn){
	feature_sz.push_back(cn_features.data_sz_block1);
	feature_dim.push_back(cn_features.fparams.nDim);
	compressed_dim.push_back(cn_features.fparams.compressed_dim);
    }

    feature_sz.push_back(hog_features.data_sz_block1);
    feature_dim.push_back(hog_features.fparams.nDim);
    compressed_dim.push_back(hog_features.fparams.compressed_dim);

    output_sz = 0;
    for (size_t i = 0; i != feature_sz.size(); ++i)
    {
        size_t size = feature_sz[i].width + (feature_sz[i].width + 1) % 2 ;
        //13*13 53*53 59*59
        filter_sz.push_back(cv::Size(size, size));
        k1 = size > output_sz ? i : k1;
        output_sz = std::max(size,output_sz);
	
    }

    for (size_t i = 0; i < filter_sz.size(); ++i)
    {
        cv::Mat_<float> tempy(filter_sz[i].height, 1, CV_32FC1);
        cv::Mat_<float> tempx(1, filter_sz[i].height / 2 + 1, CV_32FC1);

        for (int j = 0; j < tempy.rows; j++)
        {
            tempy.at<float>(j, 0) = j - (tempy.rows / 2); 
        }
        ky.push_back(tempy);
        
        float* tempxData = tempx.ptr<float>(0);
        for (int j = 0; j < tempx.cols; j++)
        {
            tempxData[j] = j - (filter_sz[i].height/2);
        }
        kx.push_back(tempx);
    }

    yf_gaussion();
    cos_wind();
    for (size_t i = 0; i < filter_sz.size(); ++i)
    {
        cv::Mat interp1_fs1, interp2_fs1;
        interpolator::get_interp_fourier(filter_sz[i], interp1_fs1, interp2_fs1, params.interpolation_bicubic_a);
        interp1_fs.push_back(interp1_fs1);
        interp2_fs.push_back(interp2_fs1);
    }

    for (size_t i = 0; i < filter_sz.size(); i++)
    {
        cv::Mat temp = get_reg_filter(img_support_sz, base_target_sz, params);
        reg_filter.push_back(temp);
        cv::Mat_<float> t = temp.mul(temp);
        float energy = FFTTools::mat_sum(t);
        reg_energy.push_back(energy);
    }

    if (params.use_scale_filter)
	fdsst.init_scale_filter(params);

    int half_scales = ( params.number_of_scales - 1) /2 ;

    for (int i = -half_scales; i < half_scales +1 ; i++)
          scaleFactors.push_back(pow(params.scale_step, i));


    cv::Point sample_pos = cv::Point(round(pos.x),round(pos.y));

    ECO_FEATS xl, xlw, xlf, xlf_porj;

    feat_extrator.init(params);
    xl = feat_extrator.extractor(im, sample_pos, vector<float>(1, currentScaleFactor));


    xlf = do_windows_x(xl, cos_window);


    xlf = do_dft(xlf);

    xlf = interpolate_dft(xlf, interp1_fs, interp2_fs);

    xlf = compact_fourier_coeff(xlf);

    cv::Point2f shift_samp = cv::Point2f(pos - cv::Point2f(sample_pos));
    shift_samp = shift_samp * 2 * CV_PI * (1 / (currentScaleFactor * img_support_sz.width));


    xlf = shift_sample(xlf, shift_samp, kx, ky);
   
    projection_matrix = init_projection_matrix(xl, compressed_dim, feature_dim);


    xlf_porj = project_sample(xlf, projection_matrix);
    

    SampleUpdate.init(filter_sz, compressed_dim,params);

    SampleUpdate.update_sample_sapce_model(xlf_porj);
  
    
    ECO_FEATS new_sample_energy = feats_pow2(xlf_porj);
    
    sample_energy = new_sample_energy;

    vector<cv::Mat> proj_energy = project_mat_energy(projection_matrix, yf);


    ECO_FEATS hf, hf_inc;
        
    for (size_t i = 0; i < xlf.size(); i++)
    {
        hf.push_back(vector<cv::Mat>(xlf_porj[i].size(), cv::Mat::zeros(xlf_porj[i][0].size(), CV_32FC2)));
        hf_inc.push_back(vector<cv::Mat>(xlf_porj[i].size(), cv::Mat::zeros(xlf_porj[i][0].size(), CV_32FC2)));
    }

#ifdef  ECO_TRAIN

    eco_trainer.train_init(hf, hf_inc, projection_matrix, xlf, yf, reg_filter,
        new_sample_energy, reg_energy, proj_energy, params);

    eco_trainer.train_joint();

    projection_matrix = eco_trainer.get_proj();
   
    xlf_porj = project_sample(xlf, projection_matrix);

#endif
    SampleUpdate.replace_sample(xlf_porj, 0);

    float new_sample_norm = FeatEnergy(xlf_porj);    

 
    SampleUpdate.set_gram_matrix(0, 0, 2 * new_sample_norm);

    frames_since_last_train = 0;

#ifdef  ECO_TRAIN
    hf_full = full_fourier_coeff((eco_trainer.get_hf()));

#endif

    if (params.use_scale_filter)
    	fdsst.scale_filter_update(im, pos, base_target_sz, currentScaleFactor);
    if (debug){

    cout << "img_support_sz is : " << img_support_sz << std::endl;
    cout << "target_sz " << target_sz <<endl;

    cout << "base_target_sz " << base_target_sz <<endl;
    cout << "search_area " << search_area  <<endl;
    cout << "scaleFactors.size " << scaleFactors.size()  <<endl;

    cout <<  "img_sample_sz" <<  img_sample_sz << endl;
    //cout << "scale_step " << scale_step  <<endl;
    cout << "output_sz " << output_sz  <<endl;

    cout << "filter_sz[0] " << filter_sz[0] << endl;
    cout << "filter_sz[1] " << filter_sz[1] << endl;
    cout << "filter_sz[2] " << filter_sz[2] << endl;
    //cout << "yf[0] "  << yf[0] << endl;
    //cout << "interp1_fs[0] "  << interp1_fs[0] << endl;
    //cout << "interp2_fs[0] "  << interp2_fs[0] << endl;
    //cout << "cos_window[0]" << cos_window[0] << endl;

    cout << "reg_filter[0] " << reg_filter[0] <<endl;
    cout << "reg_energy[0]" <<  reg_energy[0] << endl;
    cout << "currentScaleFactor " <<currentScaleFactor<<endl;
    cout <<  "pos "  <<  pos <<endl;

    }
}



     
void ECO::init_features()
{
	vector<int> cell_size;
	int max_cell_size ;
	cell_size.push_back(hog_features.fparams.cell_size);
	if (cn_features.fparams.use_cn)
		cell_size.push_back(cn_features.fparams.cell_size);

        max_cell_size = hog_features.fparams.cell_size;
        
        int new_sample_sz = (1 + 2 *round(float( img_sample_sz.width) /float (2 * max_cell_size))) * max_cell_size;
	

        int max_odd = -100, max_idx = -1; 

        for (int i = 0; i < max_cell_size; i++)
        {
	    int num_odd_dimensions = 0;
	    for (int j = 0; j < cell_size.size(); j++)
	    { 
                int sz_ = (new_sample_sz + i) / cell_size[j];
		num_odd_dimensions+=sz_ % 2 ; 
	    }
	    if (num_odd_dimensions > max_odd)
	    {
		max_odd = num_odd_dimensions;
		max_idx =  i ;
	    }
        }
        new_sample_sz += max_idx;
        img_support_sz = cv::Size(new_sample_sz, new_sample_sz);

        hog_features.img_sample_sz = img_support_sz;
        hog_features.img_input_sz = img_support_sz;

        hog_features.data_sz_block1 = cv::Size(img_support_sz.width / hog_features.fparams.cell_size, img_support_sz.height / hog_features.fparams.cell_size);
        ECO::img_support_sz = img_support_sz;
	if (cn_features.fparams.use_cn)
	    cn_features.data_sz_block1 = cv::Size(img_support_sz.width / cn_features.fparams.cell_size, img_support_sz.height / cn_features.fparams.cell_size);

        params.hog_feat = hog_features;  
}




void ECO::process_frame(const cv::Mat& frame)
    {


        cv::Point sample_pos = cv::Point(pos);
        vector<float> det_samples_pos;
        for (size_t i = 0; i < scaleFactors.size(); ++i)
        {
            det_samples_pos.push_back(currentScaleFactor * scaleFactors[i]);
        }
        // 提取特征
        ECO_FEATS xt = feat_extrator.extractor(frame, sample_pos, det_samples_pos );

        //2:  project sample *****
        ECO_FEATS xt_proj = FeatProjMultScale(xt, projection_matrix);

        // 余弦窗
        xt_proj = do_windows_x(xt_proj, cos_window);

        // 傅里叶变换
        xt_proj = do_dft(xt_proj);

        // 插值
        xt_proj = interpolate_dft(xt_proj, interp1_fs, interp2_fs);


        // 5: compute the scores of different scale of target
        //vector<cv::Mat> scores_fs_sum(scaleFactors.size(), cv::Mat::zeros(filter_sz[k1], CV_32FC2));
	//滤波
        vector<cv::Mat> scores_fs_sum;
        for (size_t i = 0; i < scaleFactors.size(); i++)
            scores_fs_sum.push_back(cv::Mat::zeros(filter_sz[k1], CV_32FC2));

        for (size_t i = 0; i < xt_proj.size(); i++)
        {
            int pad = (filter_sz[k1].height - xt_proj[i][0].rows) / 2;
            cv::Rect roi = cv::Rect(pad, pad, xt_proj[i][0].cols, xt_proj[i][0].rows);
            for (size_t j = 0; j < xt_proj[i].size(); j++)
            {
                cv::Mat score = complexMultiplication(xt_proj[i][j], hf_full[i][j % hf_full[i].size()]);
                score += scores_fs_sum[j / hf_full[i].size()](roi);
                score.copyTo(scores_fs_sum[j / hf_full[i].size()](roi));
            }
        }


        // 6: Locate the positon of target 
        optimize_scores scores(scores_fs_sum, params.newton_iterations);
        scores.compute_scores();
        float dx, dy;
        int scale_change_factor;
        scale_change_factor = scores.get_scale_ind();
	
        //scale_change_factor = 2;   // remember to delete , just for tets debug!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        dx = scores.get_disp_col() * (img_support_sz.width / output_sz) * currentScaleFactor * scaleFactors[scale_change_factor];
        dy = scores.get_disp_row() * (img_support_sz.height / output_sz) * currentScaleFactor * scaleFactors[scale_change_factor];
        //cv::Point old_pos;
        pos = cv::Point2f(sample_pos) + cv::Point2f(dx, dy);

	if (params.use_scale_filter)
		currentScaleFactor *= fdsst.scale_filter_track(frame, pos, base_target_sz, currentScaleFactor);

        currentScaleFactor = currentScaleFactor *  scaleFactors[scale_change_factor];
        vector<float> sample_scale;
        for (size_t i = 0; i < scaleFactors.size(); ++i)
        {
            sample_scale.push_back(scaleFactors[i] * currentScaleFactor);
        }

        //*****************************************************************************
        //*****                     Model update step
        //******************************************************************************

        // 1: Use the sample that was used for detection
        ECO_FEATS xtlf_proj;
	//取scale_change_factor处的所有特征的一半行
        for (size_t i = 0; i < xt_proj.size(); ++i)
        {
            std::vector<cv::Mat> tmp;
            int start_ind = scale_change_factor      *  projection_matrix[i].cols;
            int end_ind = (scale_change_factor + 1)  *  projection_matrix[i].cols;
            for (size_t j = start_ind; j < end_ind; ++j)
            {
                tmp.push_back(xt_proj[i][j].colRange(0, xt_proj[i][j].rows / 2 + 1));
            }
            xtlf_proj.push_back(tmp);
        }

        // 2: cv::Point shift_samp = pos - sample_pos : should ba added later !!!
        cv::Point2f shift_samp = cv::Point2f(pos - cv::Point2f(sample_pos));
        shift_samp = shift_samp * 2 * CV_PI * (1 / (currentScaleFactor * img_support_sz.width));
        xtlf_proj = shift_sample(xtlf_proj, shift_samp, kx, ky);

        // 3: Update the samplesf new sample, distance matrix, kernel matrix and prior weight
	//更新sample
        SampleUpdate.update_sample_sapce_model(xtlf_proj);

        // 4: insert new sample

	//miaopass edit
        /*if (SampleUpdate.get_merge_id() > 0)
        {
            SampleUpdate.replace_sample(xtlf_proj, SampleUpdate.get_merge_id());
        }
        if (SampleUpdate.get_new_id() > 0)
        {
            SampleUpdate.replace_sample(xtlf_proj, SampleUpdate.get_new_id());
        }
*/
        // 5: update filter parameters 
        bool train_tracker = frames_since_last_train >= params.train_gap;
        if (train_tracker)
        {
            ECO_FEATS new_sample_energy = feats_pow2(xtlf_proj);
            sample_energy = FeatScale(sample_energy, 1 - params.learning_rate) + FeatScale(new_sample_energy, params.learning_rate);
            eco_trainer.train_filter(SampleUpdate.get_samples(), SampleUpdate.get_samples_weight(), sample_energy);
            frames_since_last_train = 0;
        }
        else
        {
            ++frames_since_last_train;
        }
        projection_matrix = eco_trainer.get_proj();//*** exect to matlab tracker
        hf_full = full_fourier_coeff(eco_trainer.get_hf());

        if (params.use_scale_filter)
    	    fdsst.scale_filter_update(frame, pos, base_target_sz, currentScaleFactor);



        //*****************************************************************************
        //*****                    just for test
        //******************************************************************************
	if (params.visualization)
	{
        	cv::Rect resbox;
        	resbox.width = base_target_sz.width * currentScaleFactor;
        	resbox.height = base_target_sz.height * currentScaleFactor;
        	resbox.x = pos.x - resbox.width / 2;
        	resbox.y = pos.y - resbox.height / 2;

        	cv::Mat resframe = frame.clone();
        	cv::rectangle(resframe, resbox, cv::Scalar(0, 255, 0));
        	cv::imshow("ECO-Tracker", resframe);
        	cv::waitKey(1);
	}
}



void ECO::yf_gaussion()
    {
        float sig_y = sqrt(int(base_target_sz.width)*int(base_target_sz.height))*
        (params.output_sigma_factor)*(float(output_sz) / img_support_sz.width);
        for (int i = 0; i < ky.size(); i++)
        {
            // ***** opencv matrix operation ******
            cv::Mat tempy(ky[i].size(), CV_32FC1);
            tempy = CV_PI * sig_y * ky[i] / output_sz;
            cv::exp(-2 * tempy.mul(tempy), tempy);
            tempy = sqrt(2 * CV_PI) * sig_y / output_sz * tempy;
	    
            cv::Mat tempx(kx[i].size(), CV_32FC1);
            tempx = CV_PI * sig_y * kx[i] / output_sz;
            cv::exp(-2 * tempx.mul(tempx), tempx);
            tempx = sqrt(2 * CV_PI) * sig_y / output_sz * tempx;

            yf.push_back(cv::Mat(tempy * tempx));        //*** hehe  over ****
         }
    }




void ECO::cos_wind()
{


    for (size_t i = 0; i < feature_sz.size(); i++)
        {
        cv::Mat hann1t = cv::Mat(cv::Size(feature_sz[i].width + 2, 1), CV_32F, cv::Scalar(0));
        cv::Mat hann2t = cv::Mat(cv::Size(1, feature_sz[i].height + 2), CV_32F, cv::Scalar(0));

        for (int i = 0; i < hann1t.cols; i++)
            hann1t.at<float >(0, i) = 0.5 * (1 - std::cos(2 * CV_PI * i / (hann1t.cols  - 1)));
        for (int i = 0; i < hann2t.rows; i++)
            hann2t.at<float >(i, 0) = 0.5 * (1 - std::cos(2 * CV_PI * i / (hann2t.rows - 1)));
        cv::Mat hann2d = hann2t * hann1t;
        cos_window.push_back(hann2d(cv::Range(1, hann2d.rows - 1), cv::Range(1, hann2d.cols - 1)));
    }//end for 
}




ECO_FEATS  ECO::do_windows_x(const ECO_FEATS& xl, vector<cv::Mat>& cos_win)
    {
        ECO_FEATS xlw;
        for (size_t i = 0; i < xl.size(); i++)
        {
            vector<cv::Mat> temp;
		
            for (size_t j = 0; j < xl[i].size(); j++){
		
                temp.push_back(cos_win[i].mul(xl[i][j]));
		}
            xlw.push_back(temp);
        }
        return xlw;
}


ECO_FEATS  ECO::interpolate_dft(const ECO_FEATS& xlf, vector<cv::Mat>& interp1_fs, vector<cv::Mat>& interp2_fs)
{
        ECO_FEATS result;
	//cout << "row" << interp1_fs[0] << endl;
	//cout << "col" << interp2_fs[0] << endl;
        for (size_t i = 0; i < xlf.size(); i++)
        {
            cv::Mat interp1_fs_mat = RectTools::subwindow(interp1_fs[i], cv::Rect(cv::Point(0, 0), cv::Size(interp1_fs[i].rows, interp1_fs[i].rows)), IPL_BORDER_REPLICATE);
            cv::Mat interp2_fs_mat = RectTools::subwindow(interp2_fs[i], cv::Rect(cv::Point(0, 0), cv::Size(interp2_fs[i].cols, interp2_fs[i].cols)), IPL_BORDER_REPLICATE);
            vector<cv::Mat> temp;
	    
            for (size_t j = 0; j < xlf[i].size(); j++)
            {
                temp.push_back(complexMultiplication(complexMultiplication(interp1_fs_mat, xlf[i][j]), interp2_fs_mat));

            }
            result.push_back(temp);
        }
        return result;
}




ECO_FEATS  ECO::compact_fourier_coeff(const ECO_FEATS& xf)
{
        ECO_FEATS result;
        for (size_t i = 0; i < xf.size(); i++)
        {
            vector<cv::Mat> temp;
            for (size_t j = 0; j < xf[i].size(); j++)
                temp.push_back(xf[i][j].colRange(0, (xf[i][j].cols + 1) / 2));
            result.push_back(temp);
        }
        return result;
}




vector<cv::Mat> ECO::init_projection_matrix(const ECO_FEATS& init_sample, const vector<int>& compressed_dim, const vector<int>& feature_dim)
    {
        vector<cv::Mat> result;
        for (size_t i = 0; i < init_sample.size(); i++)
        {
            cv::Mat feat_vec(init_sample[i][0].size().area(), feature_dim[i], CV_32FC1);
            //cv::Mat mean(init_sample[i][0].size().area(), feature_dim[i], CV_32FC1);
            for (int j = 0; j < init_sample[i].size(); j++)
            {
                float mean = cv::mean(init_sample[i][j])[0];
		//cout << "mean " << mean << endl;
                for (size_t r = 0; r < init_sample[i][j].rows; r++)
                    for (size_t c = 0; c < init_sample[i][j].cols; c++)
                        feat_vec.at<float>(c * init_sample[i][j].rows + r, j) = init_sample[i][j].at<float>(r, c) - mean;
            }
		
            result.push_back(feat_vec);
        }
        
        vector<cv::Mat> proj_mat;
        //****** svd operation ******
        for (size_t i = 0; i < result.size(); i++)
        {
            cv::Mat S, V, D;
            cv::SVD::compute(result[i].t()*result[i], S, V, D);
            vector<cv::Mat> V_;
            V_.push_back(V); V_.push_back(cv::Mat::zeros(V.size(), CV_32FC1));
            cv::merge(V_, V);
	    //miaopass edit
            proj_mat.push_back(-V.colRange(0, compressed_dim[i]));  //** two channels : complex 
        }

        return proj_mat;
}




vector<cv::Mat> ECO::project_mat_energy(vector<cv::Mat> proj, vector<cv::Mat> yf)
{
        vector<cv::Mat> result;

        for (size_t i = 0; i < yf.size(); i++)
        {
            cv::Mat temp(proj[i].size(), CV_32FC1), temp_compelx;
            float sum_dim = std::accumulate(feature_dim.begin(), feature_dim.end(), 0);
            cv::Mat x = yf[i].mul(yf[i]);

            temp = 2 * FFTTools::mat_sum(x) / sum_dim * cv::Mat::ones(proj[i].size(), CV_32FC1);
		
            result.push_back(temp);
        }
        return result;
}




ECO_FEATS  ECO::full_fourier_coeff(ECO_FEATS xf)
{    
        ECO_FEATS res;
        for (size_t i = 0; i < xf.size(); i++)
        {
            vector<cv::Mat> tmp;
            for (size_t j = 0; j < xf[i].size(); j++)
            {
                cv::Mat temp = xf[i][j].colRange(0, xf[i][j].cols - 1).clone();
                rot90(temp, 3);
                cv::hconcat(xf[i][j], mat_conj(temp), temp);
                tmp.push_back(temp);
            }
            res.push_back(tmp);
        }

        return res;
}


ECO_FEATS ECO::shift_sample(ECO_FEATS& xf, cv::Point2f shift, std::vector<cv::Mat> kx, std::vector<cv::Mat>  ky)
{
        ECO_FEATS res;

        for (size_t i = 0; i < xf.size(); ++i)
        {
            cv::Mat shift_exp_y(ky[i].size(), CV_32FC2), shift_exp_x(kx[i].size(), CV_32FC2);
            for (size_t j = 0; j < ky[i].rows; j++)
            {
                shift_exp_y.at<COMPLEX>(j, 0) = COMPLEX(cos(shift.y * ky[i].at<float>(j, 0)), sin(shift.y *ky[i].at<float>(j, 0)));
            }

            for (size_t j = 0; j < kx[i].cols; j++)
            {
                shift_exp_x.at<COMPLEX>(0, j) = COMPLEX(cos(shift.x * kx[i].at<float>(0, j)), sin(shift.x * kx[i].at<float>(0, j)));
            }

            cv::Mat shift_exp_y_mat = RectTools::subwindow(shift_exp_y, cv::Rect(cv::Point(0, 0), xf[i][0].size()), IPL_BORDER_REPLICATE);
            cv::Mat shift_exp_x_mat = RectTools::subwindow(shift_exp_x, cv::Rect(cv::Point(0, 0), xf[i][0].size()), IPL_BORDER_REPLICATE);

            vector<cv::Mat> tmp;
            for (size_t j = 0; j < xf[i].size(); j++)
            {
                tmp.push_back(complexMultiplication(complexMultiplication(shift_exp_y_mat, xf[i][j]), shift_exp_x_mat));
            }
            res.push_back(tmp);
        }
        return res;
}
