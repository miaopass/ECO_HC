#include "fDSST.h"

using namespace std;
void fDSST::init_scale_filter(eco_params& pparams)
{
	
	params = pparams;

	init_target_sz = params.init_sz;
	
	nScales = params.number_of_scales_filter;
	scale_step = params.scale_step_filter;

	scale_sigma = params.number_of_interp_scales * params.scale_sigma_factor;

	cv::Mat scale_exp(1 , nScales , CV_32FC1);
	for(int i = 0; i < nScales; i++)
		scale_exp.at<float>(0,i) = float(i - (nScales - 1 ) / 2 ) * params.number_of_interp_scales / nScales ;
	cv::Mat scale_exp_shift(1 , nScales , CV_32FC1);
	for (int i = 0; i < nScales ; i++)
		if (i < (nScales + 1) / 2)
			scale_exp_shift.at<float>(0,i) = scale_exp.at<float>(0,i + (nScales - 1) / 2);
		else
			scale_exp_shift.at<float>(0,i) = scale_exp.at<float>(0,i - (nScales + 1) / 2);

	cv::Mat temp_scaleSizeFactors(1 , nScales , CV_32FC1);
	
	for (int i = 0; i < nScales ; i++)
		temp_scaleSizeFactors.at<float>(0,i) = pow(scale_step , scale_exp.at<float>(0,i));
	scaleSizeFactors = temp_scaleSizeFactors;
	cv::Mat temp_interpScaleFactors(1 , params.number_of_interp_scales , CV_32FC1);

	for(int i = 0; i < params.number_of_interp_scales; i++)
		temp_interpScaleFactors.at<float>(0,i) = pow(scale_step , i + i * 2 / (params.number_of_interp_scales + 1) * (- params.number_of_interp_scales) );
	interpScaleFactors = temp_interpScaleFactors;
	cv::Mat ys(1 , nScales , CV_32FC1);
	for (int i = 0; i < nScales ; i++)
		ys.at<float>(0,i) = exp(-0.5f * pow(scale_exp_shift.at<float>(0,i) , 2) / pow(scale_sigma,2) );
	
	cv::Mat temp_yf = FFTTools::fftd(ys);
	yf = temp_yf;
	if ( pow(params.scale_model_factor, 2) * init_target_sz.area() > params.scale_model_max_area )
		params.scale_model_factor = sqrt( float(params.scale_model_max_area) / init_target_sz .area() ) ;


	scale_model_sz.width  =  max( 8, (int) (params.scale_model_factor * init_target_sz.width ));
	scale_model_sz.height  =  max( 8, (int)(params.scale_model_factor * init_target_sz.height ));
	//int s_num_compressed_dim = scaleSizeFactors.cols;
	//int scaleFactors = 1;
	//params.scale_model_sz = init_target_sz ;


	//cv::Size init_target_sz = params.init_sz
        cv::Mat hann1t(nScales,nScales, CV_32FC1, cv::Scalar(0));

        for (int i = 0; i < hann1t.cols; i++)
            hann1t.at<float >(0, i) = 0.5 * (1 - std::cos(2 * CV_PI * i / (hann1t.cols  - 1)));
	for (int i = 1; i < hann1t.cols; i++)
	    hann1t.row(0).copyTo(hann1t.row(i));
        cos_window = hann1t;

}
int sumtime1 =0;
void  fDSST::scale_filter_update(cv::Mat im, cv::Point2f pos, cv::Size2f base_target_sz, float currentScaleFactor)
{
	cv::Mat scales = currentScaleFactor*scaleSizeFactors;

	cv::Mat xs = extract_scale_sample(im ,pos ,base_target_sz ,scales);

	if (s_num.empty())
		s_num = xs;	
	else
		s_num  = (1 - params.scale_learning_rate) * s_num + params.scale_learning_rate * xs;
	


	vector<cv::Mat> temp_basis  = ypQR_Schmidt(s_num);
        //temp_basis = temp_basis * s_num.t();
	vector<cv::Mat> scale_basis_den  = ypQR_Schmidt(xs); 


	//scale_basis_den = scale_basis_den * xs.t();
	//cv::Mat S, temp_basis, D , scale_basis_den;

	//cv::SVD::compute( s_num, S, temp_basis, D);
	//cv::SVD::compute( xs, S, scale_basis_den, D);

	temp_basis[0].copyTo(basis);


	//cv::transpose(scale_basis_den[0] , scale_basis_den[0]);
	//temp_basis[0].copyTo(basis); 


	cv::Mat sf_proj = FFTTools::fftr(cos_window.mul(temp_basis[1])) ;
	cv::Mat sf_num_temp = FFTTools::complexbsxfun(FFTTools::mat_conj(sf_proj),yf);
	sf_num_temp.copyTo(sf_num);

	

	cv::Mat xsf = FFTTools::fftr(cos_window.mul(scale_basis_den[1])) ;
	cv::Mat new_sf_den = FFTTools::eng(xsf);
	cv::reduce(new_sf_den,new_sf_den,0,CV_REDUCE_SUM);

	
	
	



	if (!sf_den.empty())
		new_sf_den = (1 - params.scale_learning_rate) * sf_den + params.scale_learning_rate * new_sf_den;
	new_sf_den.copyTo(sf_den);

	
/*just for test
	CvMat* cvMat3 = R;
	cv::Mat del = cv::cvarrToMat(cvMat3);
	cv::Mat test(17,17,del.type());
	for (int i=0;i<17;i++)
		 del.row(i).copyTo(test.row(i));
	cv::Mat res = basis * test - s_num;
	float a = 0.0f;
	for (int i = 0 ;i < res.rows;i++)
		for (int j = 0 ;j < res.cols;j++)
			if (a < res.at<float>(i,j))
				a = res.at<float>(i,j);
	cout <<"a "  <<  a << endl;
*/

}

float  fDSST::scale_filter_track(cv::Mat im, cv::Point2f pos, cv::Size2f base_target_sz, float currentScaleFactor)
{

	cv::Mat scales = currentScaleFactor * scaleSizeFactors;
	
	cv::Mat xs = extract_scale_sample(im ,pos ,base_target_sz ,scales);

	cv::Mat xsf = FFTTools::fftr(cos_window.mul(basis * xs)) ;
	
	cv::Mat scale_responsef = FFTTools::complexMultiplication(sf_num,xsf) ;
	cv::reduce(scale_responsef ,scale_responsef ,0 ,CV_REDUCE_SUM);
	scale_responsef = FFTTools::complexDivisionreal(scale_responsef ,sf_den + params.lambda);
	scale_responsef = resizeDFT(scale_responsef ,params.number_of_interp_scales);
	cv::dft(scale_responsef,scale_responsef,(cv::DFT_INVERSE | cv::DFT_SCALE));
	scale_responsef =  FFTTools::real(scale_responsef);
	cv::Point recovered_scale_index ;
	minMaxLoc(scale_responsef,NULL,NULL,NULL,&recovered_scale_index);
	int index = recovered_scale_index.x;

		int id1 = (index -1 ) % params.number_of_interp_scales ;
		if (id1 < 0)
			id1 += params.number_of_interp_scales;
		int id2 = (index +1 ) % params.number_of_interp_scales ;
		cv::Mat poly_x(3,1,CV_32FC1) , poly_y(3,1,CV_32FC1);
		poly_x.at<float>(0,0) = interpScaleFactors.at<float>(0,id1);
		poly_x.at<float>(1,0) = interpScaleFactors.at<float>(0,index);
		poly_x.at<float>(2,0) = interpScaleFactors.at<float>(0,id2);

		poly_y.at<float>(0,0) = scale_responsef.at<float>(0,id1);
		poly_y.at<float>(1,0) = scale_responsef.at<float>(0,index);
		poly_y.at<float>(2,0) = scale_responsef.at<float>(0,id2);	
	
		cv::Mat poly_A_mat(3,3,CV_32FC1,cv::Scalar(1));
		poly_x.copyTo(poly_A_mat.col(1));
		poly_x = poly_x.mul(poly_x);
		poly_x.copyTo(poly_A_mat.col(0));
		
		cv::Mat poly =  poly_A_mat.inv() * poly_y;

		float scale_change_factor = - poly.at<float>(0,1) / (2 *  poly.at<float>(0,0));
		
		
	return  scale_change_factor;
	
}

cv::Mat fDSST::resizeDFT(cv::Mat inputdft, int desiredLen)
{

	int minsz = min(desiredLen , nScales);
	
	float scaling = float(desiredLen)/nScales;
	cv::Mat res(1, desiredLen , inputdft.type(),cv::Scalar(0));
	int mids = minsz/2 + 1;
	int mide = int((minsz-1)/2) + 1 ;
	for(int i =0 ; i < mids ; i++)
		res.at<cv::Vec<float, 2> >(0, i) = scaling * inputdft.at<cv::Vec<float, 2> >(0, i);
	for(int i =1 ; i < mide ; i++)
		res.at<cv::Vec<float, 2> >(0, desiredLen - i) = scaling * inputdft.at<cv::Vec<float, 2> >(0, nScales - i);

	return res;
}

cv::Mat fDSST::extract_scale_sample(cv::Mat im, cv::Point2f pos, cv::Size2f base_target_sz, cv::Mat scaleFactors)
{

	double  ddf = 0.0;
	cv::minMaxLoc(scaleFactors, &ddf,NULL,NULL,NULL);
	int df  = (int) ddf;
	cv::Mat new_im;
	cv::Point2f new_pos;
	cv::Mat new_scaleFactors;
	cv::Mat res;
	if (df > 1)
	{	
		new_im.create((im.rows-1)/df + 1,(im.cols -1)/df + 1, im.type());
		for (int i = 0 ; i < new_im.rows; i++)
			for (int j = 0; j < new_im.cols; j++)
				new_im.at<cv::Vec3b>(i,j) = im.at<cv::Vec3b>(i*df,j*df);
		pos.x = (pos.x - 1) /df + 1;
		pos.y = (pos.y - 1) /df + 1;
		scaleFactors = scaleFactors / df;
	}
	else
		new_im = im;
	int dim_scale = 0 ;


	HogFeature scale_fhog(4, 1);
	for (int i = 0; i < nScales; i++)
	{
		cv::Size patch_sz;
		patch_sz.width =(int)(base_target_sz.width * scaleFactors.at<float>(0,i));
		patch_sz.height =(int)(base_target_sz.height * scaleFactors.at<float>(0,i));
		cv::Mat im_patch(patch_sz.height,patch_sz.width,im.type());
		for (int ii = 0; ii < im_patch.rows; ii++)
		{
			for (int jj = 0; jj < im_patch.cols; jj++)
			{
				int aa = (int)pos.y + ii - (int)(im_patch.rows/2);
				int bb = (int)pos.x + jj - (int)(im_patch.cols/2);
				if (aa < 0)
					aa = 0;
				else if (aa >= new_im.rows)
					aa = new_im.rows - 1;
				if (bb <0)
					bb = 0;
				else if (bb >= new_im.cols )
					bb = new_im.cols - 1;
				im_patch.at<cv::Vec3b>(ii,jj) = new_im.at<cv::Vec3b>(aa,bb);			
			}
		}

		cv::Mat im_patch_resized;
		if (scale_model_sz.width > im_patch.rows)
			cv::resize(im_patch, im_patch_resized, scale_model_sz, 0, 0 ,CV_INTER_LINEAR);
		else
			cv::resize(im_patch, im_patch_resized, scale_model_sz, 0, 0 ,CV_INTER_AREA);

		cv::Mat feat_hog = scale_fhog.getFeature(im_patch_resized);


		if (i == 0)
		{
			dim_scale = feat_hog.cols*feat_hog.rows*feat_hog.channels();
			//cv::Mat scale_sample(dim_scale , nScales , CV_32FC1 , cv::Scalar(0));
			//res = scale_sample;
			res.create(dim_scale , nScales , CV_32FC1 );
		}
		feat_hog = feat_hog.reshape(1,dim_scale); 
		feat_hog.copyTo(res.col(i));


	}

	return res;
	
}
