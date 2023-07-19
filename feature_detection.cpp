#pragma once
#include "feature_detection.h"
#include <opencv2/core/eigen.hpp>
#include <random>
#include <filesystem>

FeatureDetector::FeatureDetector(double ratio_thresh, int normType, bool crossCheck)
{
	detector = cv::SIFT::create();
	BFMatcher = cv::BFMatcher::create(normType, crossCheck);
	ratio_thresh_ = ratio_thresh;
}

void FeatureDetector::create_detector()
{
	detector = cv::SIFT::create();
}

void FeatureDetector::create_descriptor_matcher()
{
	BFMatcher = cv::BFMatcher::create(cv::NORM_L2,false);
}

Eigen::Matrix4d FeatureDetector::calculateRelativeOdometry(const std::vector<Pose>& absolute_odometry, size_t id, size_t step) const
{
	Eigen::Matrix4d first_position = Eigen::Matrix4d::Identity();
	Eigen::Matrix4d second_position = Eigen::Matrix4d::Identity();

	for (const auto& position : absolute_odometry)
	{
		if (position.id == id)
		{
			//std::cout << " the id: " << position.id << std::endl;
			first_position = position.pose_of_0camera;
			//std::cout << first_position << std::endl;
		}
		if (position.id == (id + step))
		{
			//std::cout << " the id: " << position.id << std::endl;
			second_position = position.pose_of_0camera;
			//std::cout << second_position << std::endl;
		}
	}
	//std::cout << "The relative odometry" << std::endl;
	auto relative_odometry = first_position.inverse() * second_position; // O_rel =  (O_t-1)^-1 * O_t // first_position.inverse() * second_position;
	//std::cout << relative_odometry << std::endl;
	return relative_odometry;
}

std::vector<std::pair<Feature, Feature>> FeatureDetector::detectAndSortFeatures(const cv::Mat& image_1, const cv::Mat& image_2, const Eigen::Matrix4d& current_odometry, size_t first_id, size_t step_size)
{
	if (!detector)
	{
		create_detector();
	}
	if (!BFMatcher)
	{
		create_descriptor_matcher();
	}
	size_t second_id = first_id + step_size;

	std::vector<cv::KeyPoint> first_keypoints, second_keypoints;
	std::vector<std::vector<cv::DMatch>> best_matches;
	cv::Mat descriptor_1, descriptor_2;
	detector->detectAndCompute(image_1, cv::noArray(), first_keypoints, descriptor_1);
	detector->detectAndCompute(image_2, cv::noArray(), second_keypoints, descriptor_2);
	BFMatcher->knnMatch(descriptor_1, descriptor_2, best_matches, 2);
	std::vector<cv::DMatch> good_matches;
	
	std::vector<std::pair<Feature, Feature>> current_feature_set;

	for (size_t i = 0; i < best_matches.size(); i++)
	{
		if (best_matches[i][0].distance < ratio_thresh_ * best_matches[i][1].distance)
		{
			good_matches.emplace_back(best_matches[i][0]);
			Feature one;
			one.image_id = first_id;
			one.feature_point = first_keypoints[best_matches[i][0].queryIdx].pt;
			one.M = Eigen::Matrix4d::Identity();

			Feature two;
			two.image_id = second_id;
			two.feature_point = second_keypoints[best_matches[i][0].trainIdx].pt;
			two.M = current_odometry;
			auto ret = one.feature_point - two.feature_point;

			Eigen::Vector2d vector(ret.x, ret.y);
			double normed_vector = vector.norm();
			current_feature_set.emplace_back(one, two);
		}
	}
	feature_sets_.emplace_back(current_feature_set);
	cv::Mat output_matches;
	return current_feature_set;
}

bool FeatureDetector::checkOdodmetry(const Eigen::Matrix4d& relative_odometry,double speed_limit, double yaw_limit, bool use_turning_odometry, bool use_straight_odometry)
{
	double lateral_displacement = relative_odometry.block<3, 1>(0, 3)(0); // 0
	double translation_1 = relative_odometry.block<3, 1>(0, 3)(1);// Up 1 
	double translation_2 = relative_odometry.block<3, 1>(0, 3)(2);// Forward 2
	double yaw_in_angle = relative_odometry.block<3, 3>(0, 0).eulerAngles(2, 1, 0).x() * (180.0 / 3.141592653589793238463);
	//std::cout << "The current yaw angle: " << yaw_in_angle << std::endl;
	//std::cout << "The previous yaw angle: " << previous_yaw_angle << std::endl;
	//std::cout << "The current lateral displacement: " << lateral_displacement << std::endl;
	//std::cout << "The previous lateral displacement: " << previous_lateral_displacement << std::endl;
	//Speed
	if (relative_odometry.block<3, 1>(0, 3).norm() < 0.01)
	{
		previous_yaw_angle = yaw_in_angle;
		previous_lateral_displacement = lateral_displacement;
		return false;
	}
	if (abs(translation_1) > 0.1)
	{
		previous_yaw_angle = yaw_in_angle;
		previous_lateral_displacement = lateral_displacement;
		return false;
	}
	if (abs(lateral_displacement) < 0.0075)
	{
		previous_yaw_angle = yaw_in_angle;
		previous_lateral_displacement = lateral_displacement;
		return false;
	}
	previous_yaw_angle = yaw_in_angle;
	previous_lateral_displacement = lateral_displacement;
	return true;
}


void FeatureDetector::saveFeatureMatchingResult(std::vector <std::pair<Feature, Feature>> one_feature_set,cv::Mat first_image, cv::Mat second_image,size_t id, size_t step_size)
{
	std::cout << "Current ID: " << id << "Step size: " << step_size << std::endl;
	cv::Mat input_first = first_image.clone();
	cv::Mat input_second = second_image.clone();
	for (size_t i = 0; i < one_feature_set.size(); i++)
	{
		//std::cout << "Base: " << one_feature_set[i].first.feature_point << " paired with ---> " << one_feature_set[i].second.feature_point << std::endl;
		cv::circle(input_first, one_feature_set[i].first.feature_point, 2, { 0,255,0 }, 2);
		cv::circle(input_second, one_feature_set[i].second.feature_point, 2, {221,160,221}, 2);
		
	}
	std::string directory_name = "images_" + std::to_string(id);
	if (!std::filesystem::is_directory(directory_name))
	{
		std::filesystem::create_directory(directory_name);
		auto name1 = (directory_name + "/image_" + std::to_string(id) + "_" + std::to_string(one_feature_set.size()) + ".png");
		cv::imwrite(name1, input_first);
	}
	auto name2 = (directory_name + "/image_" + std::to_string(id+step_size) + "_" + std::to_string(one_feature_set.size()) + ".png");
	cv::imwrite(name2, input_second);
}

//void FeatureDetector::drawDistribution(cv::Mat image, const std::string& sequence_number, int limit_in_bins)
//{
//	for (int i = 0; i < bins_.size(); i++)
//	{
//		auto middle = (bins_[i].top_left);
//		if (bins_[i].num_of_features >= limit_in_bins)
//		{
//			cv::rectangle(image, bins_[i].top_left, bins_[i].bottom_right, cv::Scalar(0, 255, 0), 2, 8, 0);
//			cv::putText(image, std::to_string(i), middle, cv::FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),2);
//		}
//		else
//		{
//			cv::rectangle(image, bins_[i].top_left, bins_[i].bottom_right, cv::Scalar(0, 0, 255), 2, 8, 0);
//			cv::putText(image, std::to_string(i), middle, cv::FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),2);
//		}
//	}
//	cv::imshow("Distribution", image);
//	cv::waitKey(1);
//	cv::imwrite("Distribution" + sequence_number + ".png", image);
//}

//void FeatureDetector::drawDistributionWithPoints(cv::Mat image, const std::string& sequence_number, int limit_in_bins)
//{
//	for (int i = 0; i < bins_.size(); i++)
//	{
//		auto middle = (bins_[i].top_left);
//		if (bins_[i].num_of_features >= limit_in_bins)
//		{
//			cv::rectangle(image, bins_[i].top_left, bins_[i].bottom_right, cv::Scalar(0, 255, 0), 2, 8, 0);
//		}
//		else
//		{
//			cv::rectangle(image, bins_[i].top_left, bins_[i].bottom_right, cv::Scalar(0, 0, 255), 2, 8, 0);
//		}
//	}
//	for (int i = 0; i < feature_sets_.size(); i++)
//	{
//		for (int j = 0; j < feature_sets_[i].size(); j++)
//		{
//			cv::circle(image, feature_sets_[i][j].second.feature_point, 2, cv::Scalar(255, 0, 0), 2);
//		}
//	}
//	cv::imshow("Distribution", image);
//	cv::waitKey(1);
//	cv::imwrite("Distribution2" + sequence_number + ".png", image);
//}

const size_t FeatureDetector::getNumberOfFeatures()
{
	number_of_features = 0;
	for (const auto& set_member : this->feature_sets_)
	{
		number_of_features += set_member.size();
		
	}
	return number_of_features;
}

const std::vector<std::vector<std::pair<Feature, Feature>>>& FeatureDetector::getMatchedFeatures()
{
	return feature_sets_;
}

