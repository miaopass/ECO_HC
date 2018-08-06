#include  "feature_extractor.h"

#include <ctime>
int sumtime = 0;
using namespace std;


vector<string> spl(string s, string delim){
  vector<string> res;
 
  int start = 0;
  int end = -1;
  //跳过首端的连续分割符
  while ((end = s.find_first_of(delim, start)) == start)
    start = end + 1;
 
  while ((end = s.find_first_of(delim, start)) != string::npos){
    res.push_back(s.substr(start, end - start));
    start = end + 1;
    //跳过连续分割符
    while ((end = s.find_first_of(delim, start)) == start)
      start = end + 1;
  }
  //最后一段
  if (start < s.size()){
    res.push_back(s.substr(start));
  }
 
  return res;
}


cv::Mat	 cn_table(  32768,10, CV_32FC1,cv::Scalar::all(0));

void feature_extractor::init(const eco_params& params)
{
	cn_features = params.cn_feat;
	useCnFeature = cn_features.fparams.use_cn;
	cn_cell_size = cn_features.fparams.cell_size;
	if (useCnFeature)
	{
		ifstream infile; 
		infile.open("/home/miaopass/d.txt");
		string s;
		string delim = "\t";
		vector<string> test ;
		int j = 0;
		while(getline(infile,s))
		{
			test = spl(s,delim);
			for(int i = 0;i < 10;i++)
			{
			cn_table.at<float>(j,i) = atof(test[i].c_str()); 
			}
			j++;

  		}
	}
	hog_features = params.hog_feat;

}

ECO_FEATS feature_extractor::extractor(cv::Mat image, cv::Point2f pos, vector<float> scales)
{
	int num_features = 1, num_scales = scales.size();
	
 
	// extract image pathes for different kinds of feautures
	vector<vector<cv::Mat> > img_samples;
	for (int i = 0; i < num_features; ++i)
	{
		vector<cv::Mat> img_samples_temp(num_scales);
		
		for (int j = 0; j < scales.size(); ++j)
		{
			cv::Size2f img_sample_sz = hog_features.img_sample_sz;
			cv::Size2f img_input_sz =hog_features.img_input_sz;
			img_sample_sz.width *= scales[j];
			img_sample_sz.height *= scales[j];
			img_samples_temp[j] = sample_patch(image, pos, img_sample_sz, img_input_sz);

		}
		img_samples.push_back(img_samples_temp);
		
	}

	// Extract image patches features(all kinds of features)
	ECO_FEATS sum_features;


	hog_feat_maps = get_hog(img_samples[img_samples.size() - 1]);
	

	hog_feat_maps = hog_feature_normalization(hog_feat_maps);

	if(useCnFeature)
	{
		cn_feat_maps = get_cn(img_samples[img_samples.size() - 1]);

		//cn_feat_maps = cn_feature_normalization(cn_feat_maps);

		sum_features.push_back(cn_feat_maps);
	}
	sum_features.push_back(hog_feat_maps);
	return sum_features;
}

cv::Mat feature_extractor::sample_patch(const cv::Mat& im, const cv::Point2f& poss, cv::Size2f sample_sz, cv::Size2f output_sz)
{
	cv::Point pos(poss.operator cv::Point());

	// Downsample factor

	float resize_factor = std::min(sample_sz.width / output_sz.width, sample_sz.height / output_sz.height);

    int df = std::max((float)floor(resize_factor - 0.1), float(1));

    cv::Mat new_im;
	im.copyTo(new_im);
	if (df > 1)
	{

        cv::Point os((pos.x - 1) % df, ((pos.y - 1) % df));
		pos.x = (pos.x - os.x - 1) / df + 1;
		pos.y = (pos.y - os.y - 1) / df + 1;

		sample_sz.width = sample_sz.width / df;
		sample_sz.height = sample_sz.height / df;

		int r = (im.rows - os.y - 1 ) / df + 1, c = (im.cols - os.x - 1 ) / df +1;
		cv::Mat new_im2(r, c, im.type()); 

		new_im = new_im2;

		for (size_t i =  os.y , m = 0; i < im.rows && m < new_im.rows; i += df, ++m)
			for (size_t j = 0 + os.x, n = 0; j < im.cols && n < new_im.cols; j += df, ++n)
				if (im.channels() == 1)
					new_im.at<uchar>(m, n) = im.at<uchar>(i, j);
				else
					new_im.at<cv::Vec3b>(m, n) = im.at<cv::Vec3b>(i, j);
	}

	// *** extract image ***
	sample_sz.width = round(sample_sz.width);
	sample_sz.height = round(sample_sz.height);
	cv::Point pos2(pos.x - floor((sample_sz.width + 1) / 2) , pos.y - floor((sample_sz.height + 1) / 2) );
	cv::Mat im_patch = RectTools::subwindow(new_im, cv::Rect(pos2, sample_sz), IPL_BORDER_REPLICATE); // cv::Rect(cv::Point(0, 0), new_im.size())

	cv::Mat resized_patch;
	cv::resize(im_patch, resized_patch, output_sz);



	return resized_patch;
}

vector<cv::Mat>  feature_extractor::get_cn(vector<cv::Mat> ims)
{
	int den = 8 , fac = 32 , offset = 0 , region_area = 16  ;
	float  maxval = 1.0f;
	vector<cv::Mat> cn_feats;
	for (int i = 0; i < ims.size(); i++)
	{
		vector<cv::Mat> color_ims;
		cv::split((ims[i]-(den/2))/den,color_ims);
		color_ims[0].convertTo(color_ims[0], CV_16UC1);
		color_ims[1].convertTo(color_ims[1], CV_16UC1);
		color_ims[2].convertTo(color_ims[2], CV_16UC1);
		
		cv::Mat index_im =offset + (color_ims[2] ) + fac * (color_ims[1]  )+ fac * fac * (color_ims[0] );



		for (int j = 0; j < 10; j++)
		{
			cv::Mat cn_im(ims[i].rows+1,ims[i].cols+1,CV_32FC1,cv::Scalar(0));
			cv::Mat cn_feat((ims[i].rows)/cn_cell_size ,(ims[i].cols)/cn_cell_size ,CV_32FC1,cv::Scalar(0));
			for (int ii = 0; ii < cn_im.rows -1;ii ++)
				for (int jj = 0; jj < cn_im.cols -1;jj ++)
					cn_im.at<float>(ii+1,jj+1) = cn_table.at<float>(index_im.at<ushort>(ii,jj),j);
/*
			for (int ii = 0; ii < cn_im.rows -1; ii++)
				cn_im.row(ii+1)+=cn_im.row(ii);
			for (int jj = 0; jj < cn_im.cols -1; jj++)
				cn_im.col(jj+1)+=cn_im.col(jj);			
			for (int ii = 0; ii < cn_feat.cols; ii++)
				for (int jj = 0; jj < cn_feat.rows; jj++)
					cn_feat.at<float>(ii,jj) = (cn_im.at<float>((ii+1)*cn_cell_size,(jj+1)*cn_cell_size) - cn_im.at<float>(ii*cn_cell_size,(jj+1)*cn_cell_size) - cn_im.at<float>((ii+1)*cn_cell_size ,jj*cn_cell_size) + cn_im.at<float>(ii*cn_cell_size,jj*cn_cell_size)) / (region_area * maxval);
*/
			for (int ii = 0; ii < cn_feat.cols; ii++)
				for (int jj = 0; jj < cn_feat.rows; jj++)
				{
					float tp = 0.0f;
					for (int iii = 0; iii < cn_cell_size; iii++)
						for (int jjj = 0; jjj < cn_cell_size; jjj++)
							tp += cn_im.at<float>(ii * cn_cell_size + iii ,jj * cn_cell_size + jjj) ;
					cn_feat.at<float>(ii,jj) = tp / (region_area * maxval);
				}
			cn_feats.push_back(cn_feat);
		}
	}

	return cn_feats;
	
}


vector<cv::Mat> feature_extractor::get_hog(vector<cv::Mat> ims)
{
	if (ims.empty())
		return vector<cv::Mat>(); 

	vector<cv::Mat> hog_feats;
	for (int i = 0; i < ims.size(); i++)
	{
		cv::Mat temp;

		HogFeature FHOG(hog_features.fparams.cell_size, 1);

		cv::Mat features = FHOG.getFeature(ims[i]);         //*** Extract FHOG features***
		
		hog_feats.push_back(features);
	} 
	return hog_feats;
}



cv::Mat feature_extractor::sample_pool(const cv::Mat& im, int smaple_factor, int stride)
{
	if (im.empty())
		return cv::Mat();
	cv::Mat new_im(im.cols / 2, im.cols / 2, CV_32FC1);
	for (size_t i = 0; i < new_im.rows; i++)
	{
		for (size_t j = 0; j < new_im.cols; j++)
			new_im.at<float>(i, j) = 0.25 * (im.at<float>(2 * i, 2 * j) + im.at<float>(2 * i, 2 * j + 1) +
			im.at<float>(2 * i + 1, 2 * j) + im.at<float>(2 * i + 1, 2 * j + 1));
	} 
	return new_im;
}

vector<cv::Mat> feature_extractor::hog_feature_normalization(vector<cv::Mat>& hog_feat_maps)
{
	vector<cv::Mat> hog_maps_vec;
	for (size_t i = 0; i < hog_feat_maps.size(); i++)
	{
		cv::Mat  temp = hog_feat_maps[i];
		//temp = temp.mul(temp);
		// float  sum_scales = cv::sum(temp)[0]; *** sum can not work !! while dimension exceeding 3
		vector<cv::Mat> temp_vec, result_vec;
		float sum = 0;
		cv::split(temp, temp_vec);

		for (int j = 0; j < temp.channels(); j++)
			sum += cv::sum(temp_vec[j].mul(temp_vec[j]))[0];
		float para = hog_features.data_sz_block1.area() *  hog_features.fparams.nDim;
		hog_feat_maps[i] /= sqrt(sum / para);
		cv::split(hog_feat_maps[i], result_vec);
		hog_maps_vec.insert(hog_maps_vec.end(), result_vec.begin(), result_vec.end());
	}

	return hog_maps_vec;
}

vector<cv::Mat> feature_extractor::cn_feature_normalization(vector<cv::Mat>& cn_feat_maps)
{
	vector<cv::Mat> cn_maps_vec;
	for (size_t i = 0; i < cn_feat_maps.size(); i++)
	{
		cv::Mat  temp = cn_feat_maps[i];
		//temp = temp.mul(temp);
		// float  sum_scales = cv::sum(temp)[0]; *** sum can not work !! while dimension exceeding 3
		vector<cv::Mat> temp_vec, result_vec;
		float sum = 0;
		cv::split(temp, temp_vec);

		for (int j = 0; j < temp.channels(); j++)
			sum += cv::sum(temp_vec[j].mul(temp_vec[j]))[0];
		float para = temp.cols * temp.rows *  temp.channels();
		cn_feat_maps[i] /= sqrt(sum / para);
		cv::split(cn_feat_maps[i], result_vec);
		cn_maps_vec.insert(cn_maps_vec.end(), result_vec.begin(), result_vec.end());
	}

	return cn_maps_vec;
}
