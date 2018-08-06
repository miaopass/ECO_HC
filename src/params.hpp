#ifndef PARAMS_H
#define PARAMS_H

#include <vector>
#include <string>


using std::vector;
using std::string;

//**** hog parameters cofiguration *****
struct hog_params
{
	int           cell_size;
	int           compressed_dim;
	int           nOrients;
	size_t        nDim; 
	float         penalty;
};

struct cn_params
{
	bool	      use_cn;
	int           cell_size;
	int           compressed_dim;
	int           nOrients;
	size_t        nDim; 
	float         penalty;
};

 

struct hog_feature
{
        hog_params      fparams;
	cv::Size	img_input_sz;	            //*** input sample size ******
	cv::Size        img_sample_sz;			    //*** the size of sample *******
	cv::Size        data_sz_block1;			    //****  hog feature *****

};

struct cn_feature
{
        cn_params	fparams;
	cv::Size	img_input_sz;	            //*** input sample size ******
	cv::Size        img_sample_sz;			    //*** the size of sample *******
	cv::Size        data_sz_block1;			    //****  hog feature *****

};

//*** ECO parameters  configuration *****
struct eco_params
{

        cn_feature cn_feat;
        hog_feature hog_feat;

	//***** img sample parameters *****
	float  search_area_scale;
	int    min_image_sample_size;
	int    max_image_sample_size;


	//***** Detection parameters *****
	int    refinement_iterations;               // Number of iterations used to refine the resulting position in a frame
	int	   newton_iterations ;                  // The number of Newton iterations used for optimizing the detection score
	bool   clamp_position;                      // Clamp the target position to be inside the image
	bool	visualization;    

	//***** Learning parameters
	float	output_sigma_factor;			    // Label function sigma
	float	learning_rate ;	 				    // Learning rate
	size_t	nSamples;                           // Maximum number of stored training samples
	string	sample_replace_strategy;            // Which sample to replace when the memory is full
	bool	lt_size;			                // The size of the long - term memory(where all samples have equal weight)
	int 	train_gap;					        // The number of intermediate frames with no training(0 corresponds to training every frame)
	int 	skip_after_frame;                   // After which frame number the sparse update scheme should start(1 is directly)
	bool	use_detection_sample;               // Use the sample that was extracted at the detection stage also for learning

	// Regularization window parameters
	bool	use_reg_window; 					// Use spatial regularization or not
	double	reg_window_min;						// The minimum value of the regularization window
	double  reg_window_edge;					// The impact of the spatial regularization
	size_t  reg_window_power;					// The degree of the polynomial to use(e.g. 2 is a quadratic window)
	float	reg_sparsity_threshold;				// A relative threshold of which DFT coefficients that should be set to zero


	// Interpolation parameters
	string  interpolation_method;				// The kind of interpolation kernel
	float   interpolation_bicubic_a;			// The parameter for the bicubic interpolation kernel
	bool    interpolation_centering;			// Center the kernel at the feature sample
	bool    interpolation_windowing;			// Do additional windowing on the Fourier coefficients of the kernel

	// Scale parameters for the translation model
	// Only used if: params.use_scale_filter = false
	size_t  number_of_scales ;					// Number of scales to run the detector
	float   scale_step;                         // The scale factor

	cv::Size    init_sz;
	

	// Scale filter parameters
	bool        use_scale_filter ;
	float	    scale_sigma_factor ;
	float       scale_learning_rate ;
	int	    number_of_scales_filter ;
	int	    number_of_interp_scales;
	float	    scale_model_factor ;
	float	    scale_step_filter ;
	int	    scale_model_max_area ;
	string	    scale_feature ;
	string	    s_num_compressed_dim;    
	float	    lambda ;					
	bool	    do_poly_interp; 

	//***  Conjugate Gradient parameters
	int     CG_iter   ;                  // The number of Conjugate Gradient iterations in each update after the first frame
	int     init_CG_iter ;            // The total number of Conjugate Gradient iterations used in the first frame
	int     init_GN_iter ;                 // The number of Gauss - Newton iterations used in the first frame(only if the projection matrix is updated)
	bool    CG_use_FR ;                 // Use the Fletcher - Reeves(true) or Polak - Ribiere(false) formula in the Conjugate Gradient
	bool    CG_standard_alpha;         // Use the standard formula for computing the step length in Conjugate Gradient
	int     CG_forgetting_rate ;	 	   // Forgetting rate of the last conjugate direction
	float   precond_data_param ;	 	   // Weight of the data term in the preconditioner
	float   precond_reg_param  ;	 	   // Weight of the regularization term in the preconditioner
	int     precond_proj_param;	 	   // Weight of the projection matrix part in the preconditioner
	
	
	double  projection_reg; 	 	       // Regularization paremeter of the projection matrix

	

};
        



#endif
