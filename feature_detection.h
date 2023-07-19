#pragma once
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <deque>
#include "parser.h"
#include "functional_impl.h"
#include "structures.h"



class FeatureDetector
{
public:
	
private:
	cv::Ptr<cv::FeatureDetector> detector;
	cv::Ptr<cv::DescriptorMatcher> BFMatcher;
public:
	// TODO: Missing whole comments and missing comment parts!
	FeatureDetector(double ratio_thresh, int normType = cv::NORM_L2, bool crossCheck = false);
	/// Detect feature points on the first and second image and match the common points. 
	/// sequence is the image sequence which was loaded from the dataset
	/// first_id is the id of the image from the image sequence where we want to find features
	/// step_size is distance from the first image to the second image.
	std::vector<std::pair<Feature, Feature>> detectAndSortFeatures(const cv::Mat& image_1, const cv::Mat& image_2, const Eigen::Matrix4d& current_odometry, size_t first_id, size_t step_size);

	bool checkOdodmetry(const Eigen::Matrix4d& relative_odometry,double speed_limit, double yaw_limit, bool use_turning_odometry = true, bool use_straight_odometry = false);
	//void gatherFeatures(const ImageSequence& sequence, Optimizer& optimizer);

	/// Save the feature matching results to pictures and write it to console
	void saveFeatureMatchingResult(std::vector <std::pair<Feature, Feature>> one_feature_set, cv::Mat first_image, cv::Mat second_image, size_t id, size_t step_size);
	/// Returns the calculated feature paires.
	const std::vector<std::vector <std::pair<Feature, Feature>>>& getMatchedFeatures();


	const size_t getNumberOfFeatures();

	Eigen::Matrix4d calculateRelativeOdometry(const std::vector<Pose>& absolute_odometry, size_t id, size_t step) const;

private:

	void create_detector();
	void create_descriptor_matcher();

private:
	std::vector<std::vector<std::pair<Feature, Feature>>> feature_sets_;

	size_t number_of_features = 0;
	double ratio_thresh_ = 0;


	double previous_yaw_angle = 0;
	double previous_lateral_displacement = 0;


};
