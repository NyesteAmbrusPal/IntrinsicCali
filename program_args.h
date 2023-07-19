#pragma once
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <Eigen/Dense>


//TODO: Replace it with yaml file and implement a parser
struct ProgramArguments
{
	std::string sequence_number = "05";
	std::string folder = "C:/Users/Pali/Desktop/Diplomamunka_otletek/online_intrinsic_cali/";


	std::string gt_dir = folder + "data_odometry_poses/dataset/poses/" + sequence_number + ".txt";
	std::string path_to_grayscale = folder + "/dataset_odometry_grey/sequences/" + sequence_number + "/image_0/";
	std::string path_to_distorted = folder + "/dataset_odometry_distorted/" + sequence_number + "/distorted_0/";
	std::string calib_dir = folder + "data_odometry_calib/dataset/sequences/" + sequence_number + "/calib.txt";

	size_t step_size = 3;
	size_t parsed_frame_number = 500; 
	size_t starting_frame = 0; 

	cv::Point tl = { 651,226 };
	cv::Point br = { 1392 - 651,512 - 226 };
	cv::Rect aidrive_ROI = cv::Rect(tl, br);

	cv::Point tl2 = { 712,40 };
	cv::Point br2= { 1226 - 0,370 - 40 };
	cv::Rect undistorted_ROI = cv::Rect(tl2, br2);

	cv::Point tl3 = { 651,226 };
	cv::Point br3 = { 1392-651,512- 226};
	cv::Rect distortion_roi = cv::Rect(tl3, br3);

	//Boundary conditions
	double minimum_speed = 0.25;
	float ratio_thresh = 0.650;
	double max_yaw_limit = 0.65; 
	int min_pixel_displacment = 10;

	int feature_limit_in_bins = 10; //3

	int feature_limit_for_distortion_calculation = 10;

	cv::Point2i bin_size = { 18,10 };
	size_t min_pair_num = 430;

	size_t min_number_of_bin = 15;
	size_t max_number_of_bin = 18;

	//Starting params
	double focal_length = 400;
	double principal_point_x = 650;
	double principal_point_y = 250;

};