#include "eco_sample_update.h"

using namespace std;
namespace eco_sample_update{

	void sample_update::init(const std::vector<cv::Size>& filter, const std::vector<int>& feature_dim,const eco_params& params)
	{

		nSamples = params.nSamples;
		learning_rate = params.learning_rate;
		minmum_sample_weight = learning_rate * pow(1.0f - learning_rate ,nSamples * 2);

		//*** distance matrix initialization memory *****
		distance_matrix.create(cv::Size(nSamples, nSamples), CV_32FC1);
		gram_matrix.create(cv::Size(nSamples, nSamples), CV_32FC1);

		for (size_t i = 0; i < distance_matrix.rows; i++)
		{
			for (size_t j = 0; j < distance_matrix.cols; j++)
			{
				distance_matrix.at<float>(i, j) = INF;
				gram_matrix.at<float >(i, j) = INF;
			}
		}

		//*** samples memory initialization *******
		for (size_t n = 0; n < nSamples; n++)
		{
			ECO_FEATS temp;
			for (size_t feat_block = 0; feat_block < feature_dim.size(); feat_block++)
			{
				std::vector<cv::Mat> single_feat;
				for (size_t i = 0; i < feature_dim[feat_block]; i++)
					single_feat.push_back(cv::Mat::zeros(cv::Size((filter[feat_block].width + 1) / 2, filter[feat_block].width), CV_32FC2));
				temp.push_back(single_feat);
			}
			samples_f.push_back(temp);
		}
		
		prior_weights.resize(nSamples);

	}

	void  sample_update::update_sample_sapce_model(ECO_FEATS& new_train_sample)
	{
		//cout << distance_matrix.row(0) <<endl;
		merged_sample_id = -1;
		new_sample_id = -1;

		//int unm_feature_blocks = new_train_sample.size();

		//*** Find the inner product of the new sample with existing samples ***
		cv::Mat gram_vector = find_gram_vector(new_train_sample);

		float new_train_sample_norm = 2 * FeatEnergy(new_train_sample);
		cv::Mat dist_vec(nSamples, 1, CV_32FC1);
		for (int i = 0; i < nSamples; i++)
		{
			float temp = new_train_sample_norm + gram_matrix.at<float>(i, i) - 2 * gram_vector.at<float>(i, 0);
			if (i < num_training_samples)
				dist_vec.at<float>(i, 0) = std::max(temp, 0.0f);
			else
				dist_vec.at<float>(i, 0) = INF;
		}
		std::vector<int> merged_sample;
		
		if (num_training_samples == nSamples) //*** if memory is full   ****
		{
		
			float min_sample_weight = INF;
			size_t min_sample_id = 0;
			findMin(min_sample_weight, min_sample_id);

			if (min_sample_weight < minmum_sample_weight)//*** If any prior weight is less than the minimum allowed weight, replace that sample with the new sample
			{

				//*** Normalise the prior weights so that the new sample gets weight as
				update_distance_matrix(gram_vector, new_train_sample_norm, min_sample_id, -1, 0, 1);
				prior_weights[min_sample_id] = 0;

				//miaopass edit
				float sum = accumulate(prior_weights.begin(), prior_weights.end(), 0.0f);
				

				for (size_t i = 0; i < nSamples; i++)
				{
					prior_weights[i] = prior_weights[i] * (1 - learning_rate) / sum;
				}

				prior_weights[min_sample_id] = learning_rate;

				//*** Set the new sample and new sample position in the samplesf****
				new_sample_id = min_sample_id;
				new_sample = new_train_sample;
				//miaopass edit
				replace_sample( new_sample ,new_sample_id);
				

			}
			else
			{

				//*** If no sample has low enough prior weight, then we either merge
				//*** the new sample with an existing sample, or merge two of the
				//*** existing samples and insert the new sample in the vacated position
				double new_sample_min_dist;
				cv::Point min_sample_id;
				cv::minMaxLoc(dist_vec, &new_sample_min_dist, 0, &min_sample_id);

				//*** Find the closest pair amongst existing samples
				cv::Mat duplicate = distance_matrix.clone();
				double existing_samples_min_dist;
				cv::Point closest_exist_sample_pair;   //*** clost location ***
				cv::minMaxLoc(real(duplicate), &existing_samples_min_dist, 0, &closest_exist_sample_pair);

				if (closest_exist_sample_pair.x == closest_exist_sample_pair.y)
					assert("distance matrix diagonal filled wrongly ");


				if (new_sample_min_dist < existing_samples_min_dist)
				{


					//miaopass edit
					for (size_t i = 0; i < prior_weights.size(); i++)
						prior_weights[i] *= (1 - learning_rate);
					merged_sample_id = min_sample_id.y;
					ECO_FEATS existing_sample_to_merge = samples_f[merged_sample_id];
					ECO_FEATS merged_sample = merge_samples(existing_sample_to_merge, new_train_sample,
						prior_weights[merged_sample_id], learning_rate, std::string("merge"));
					update_distance_matrix(gram_vector, new_train_sample_norm, merged_sample_id, -1,
						prior_weights[merged_sample_id], learning_rate);
					prior_weights[min_sample_id.y] += learning_rate;
					//miaopass edit
					replace_sample(merged_sample,merged_sample_id);



				}
				else
				{
					//miaopass edit
					for (size_t i = 0; i < prior_weights.size(); i++)
						prior_weights[i] *= (1 - learning_rate);
					if (prior_weights[closest_exist_sample_pair.x] < prior_weights[closest_exist_sample_pair.y])
						std::swap(closest_exist_sample_pair.x, closest_exist_sample_pair.y);
					
					ECO_FEATS merged_sample = merge_samples(samples_f[closest_exist_sample_pair.x], samples_f[closest_exist_sample_pair.y],prior_weights[closest_exist_sample_pair.x], prior_weights[closest_exist_sample_pair.y], std::string("Merge"));

					update_distance_matrix(gram_vector, new_train_sample_norm, closest_exist_sample_pair.x, closest_exist_sample_pair.y,prior_weights[closest_exist_sample_pair.x], prior_weights[closest_exist_sample_pair.y]);
					prior_weights[closest_exist_sample_pair.x] += prior_weights[closest_exist_sample_pair.y];
					prior_weights[closest_exist_sample_pair.y] = learning_rate;
					merged_sample_id = closest_exist_sample_pair.x;
					new_sample_id = closest_exist_sample_pair.y;
					new_sample = new_train_sample;

					//miaopass edit
					replace_sample( merged_sample ,merged_sample_id);
					replace_sample( new_sample ,new_sample_id);

				}

			}
		}     
		else  
		{
			size_t sample_position = num_training_samples;  //*** location ****
			update_distance_matrix(gram_vector, new_train_sample_norm, sample_position, -1, 0, 1);

			if (sample_position == 0)
				prior_weights[sample_position] = 1;
			else
			{
				for (size_t i = 0; i < sample_position; i++)
					prior_weights[i] *= (1 - learning_rate);
				prior_weights[sample_position] = learning_rate;
			}

			new_sample_id = sample_position;
			new_sample = new_train_sample;
			
			num_training_samples++;
			
			//miaopass edit
			replace_sample( new_sample ,new_sample_id);
		}


	//cout <<  gram_matrix.row(0) <<endl;
	}


	cv::Mat sample_update::find_gram_vector(ECO_FEATS& new_train_sample)
	{
		cv::Mat result(cv::Size(1, nSamples), CV_32FC1,cv::Scalar(INF));
		//for (size_t i = 0; i < result.rows; i++)
			//result.at<float>(i, 0) = INF;

		std::vector<float> dist_vec;

		for (size_t i = 0; i < num_training_samples; i++)
			result.at<float>(i , 0) = 2 * feat_dis_compute(samples_f[i], new_train_sample);

		return result;
	}

	float sample_update::feat_dis_compute(std::vector<std::vector<cv::Mat> >& feat1, std::vector<std::vector<cv::Mat> >& feat2)
	{
		if (feat1.size() != feat2.size())
			return 0;

		float dist = 0;
		for (size_t i = 0; i < feat1.size(); i++)
			for (size_t j = 0; j < feat1[i].size(); j++)
				dist += FFTTools::sum_conj(feat1[i][j],feat2[i][j]);
	
		return dist;
	}

	void sample_update::update_distance_matrix(cv::Mat& gram_vector, float new_sample_norm, int id1, int id2, float w1, float w2)
	{
		float alpha1 = w1 / (w1 + w2);
		float alpha2 = 1 - alpha1;

		if (id2 < 0)
		{
			float norm_id1 = gram_matrix.at<float>(id1, id1);

			//** update the matrix ***
			if (alpha1 == 0)
			{
				gram_vector.col(0).copyTo(gram_matrix.col(id1));
				cv::Mat tt = gram_vector.t();
				tt.row(0).copyTo(gram_matrix.row(id1));
				gram_matrix.at<float>(id1, id1) = new_sample_norm;
			}
			else if (alpha2 == 0)
			{
				// *** do nothing discard new sample *****
			}
			else
			{   // *** The new sample is merge with an existing sample
				cv::Mat t = alpha1 * gram_matrix.col(id1) + alpha2 * gram_vector.col(0), t_t;
				t.col(0).copyTo(gram_matrix.col(id1));
				t_t = t.t();
				t_t.row(0).copyTo(gram_matrix.row(id1));
				gram_matrix.at<float>(id1, id1) =
					pow(alpha1, 2) * norm_id1 + pow(alpha2, 2) * new_sample_norm + 2 * alpha1 * alpha2 * gram_vector.at<float>(id1);
			}
			//*** Update distance matrix *****
			cv::Mat dist_vec(nSamples, 1, CV_32FC1);
			for (int i = 0; i < nSamples; i++)
			{
				float temp = gram_matrix.at<float>(id1, id1) + gram_matrix.at<float>(i, i) - 2 * gram_matrix.at<float>(i, id1);
				dist_vec.at<float>(i, 0) = std::max(temp, 0.0f);
			}
			dist_vec.col(0).copyTo(distance_matrix.col(id1));
			cv::Mat tt = dist_vec.t();
			tt.row(0).copyTo(distance_matrix.row(id1));
			distance_matrix.at<float>(id1, id1) = INF;
		}
		else
		{
			if (alpha1 == 0 || alpha2 == 0)
				assert("wrong");

			//*** Two existing samples are merged and the new sample fills the empty **
			float norm_id1 = gram_matrix.at<float>(id1, id1);
			float norm_id2 = gram_matrix.at<float>(id2, id2);
			float ip_id1_id2 = gram_matrix.at<float>(id1, id2);

			//*** Handle the merge of existing samples **
			cv::Mat t = alpha1 * gram_matrix.col(id1) + alpha2 * gram_matrix.col(id2), t_t;
			t.col(0).copyTo(gram_matrix.col(id1));
			cv::Mat tt = t.t();
			tt.row(0).copyTo(gram_matrix.row(id1));
			gram_matrix.at<float>(id1, id1) =
				pow(alpha1, 2) * norm_id1 + pow(alpha2, 2) * norm_id2 + 2 * alpha1 * alpha2 *ip_id1_id2;
			gram_vector.at<float>(id1) =
				alpha1 * gram_vector.at<float>(id1, 0) + alpha2 * gram_vector.at<float>(id2, 0);

			//*** Handle the new sample ****
			gram_vector.col(0).copyTo(gram_matrix.col(id2));
			tt = gram_vector.t();
			tt.row(0).copyTo(gram_matrix.row(id2));
			gram_matrix.at<float>(id2, id2) = new_sample_norm;

			//*** Update the distance matrix ****
			cv::Mat dist_vec(nSamples, 1, CV_32FC1);
			std::vector<int> id;
			id.push_back(id1);
			id.push_back(id2);
			for (size_t i = 0; i < 2; i++)
			{
				for (int j = 0; j < nSamples; j++)
				{
					float temp = gram_matrix.at<float>(id[i], id[i]) + gram_matrix.at<float>(j, j) - 2 * gram_matrix.at<float>(j, id[i]);
					dist_vec.at<float>(j, 0) = max(temp, 0.0f);
				}
				dist_vec.col(0).copyTo(distance_matrix.col(id[i]));
				cv::Mat tt = dist_vec.t();
				tt.row(0).copyTo(distance_matrix.row(id[i]));
				distance_matrix.at<float>(id[i], id[i]) = INF;
			}
		}//if end

	}//function end

 	void sample_update::findMin(float& min_w, size_t& index)const
	{
		std::vector<float>::const_iterator pos = std::min_element(prior_weights.begin(), prior_weights.end());
		min_w = *pos;
		index = pos - prior_weights.begin();
		/*min_w = INF;
		index = 0;

		for (size_t i = 0; i < prior_weights.size(); i++)
		{
			index = prior_weights[i] < min_w ? i : index;
			min_w = prior_weights[i] < min_w ? prior_weights[i] : min_w;
		}*/
	}

	sample_update::ECO_FEATS sample_update::merge_samples(ECO_FEATS& sample1, ECO_FEATS& sample2, float w1, float w2, std::string sample_merge_type)
	{
		float alpha1 = w1 / (w1 + w2);
		float alpha2 = 1 - alpha1;

		if (sample_merge_type.compare( std::string("replace"))){

			return sample1;
		}
		else if (sample_merge_type.compare( std::string("merge")))
		{

			ECO_FEATS merged_sample = sample1;
			for (size_t i = 0; i < sample1.size(); i++)
				for (size_t j = 0; j < sample1[i].size(); j++)
					merged_sample[i][j] = alpha1 * sample1[i][j] + alpha2 * sample2[i][j];
			return merged_sample;
		}
		else{

			assert("Invalid sample merge type");
		}

	}

	void sample_update::replace_sample(ECO_FEATS& new_sample, size_t idx)
	{
		samples_f[idx] = new_sample;
	}


	 void sample_update::set_gram_matrix(int r, int c, float val)
	{
		gram_matrix.at<float>(r, c) = val;
	}

}
