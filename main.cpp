#pragma once
#include <iostream>
#include <utility>
#include "feature_detection.h"
#include "focal_calibrator.h"
#include "principal_calibrator.h"
#include "distortion_calibrator.h"
#include "program_args.h"

ProgramArguments args;
Eigen::Matrix3d createMatrix(double f_x, double f_y, double c_x, double c_y)
{
	Eigen::Matrix3d K_init = Eigen::Matrix3d::Identity();
	K_init(0, 0) = f_x;
	K_init(1, 1) = f_y;
	K_init(0, 2) = c_x;
	K_init(1, 2) = c_y;
	return K_init;
}
int main()
{
	auto K_init = createMatrix(args.focal_length, args.focal_length, args.principal_point_x, args.principal_point_y);
	DistrotionParams dist_params;
	ImageSequence sequence(args.path_to_distorted, args.gt_dir, args.calib_dir, args.parsed_frame_number);
	FeatureDetector detector(args.ratio_thresh);
	FocalCalibrator focal_length_calibrator(args.undistorted_ROI, args.feature_limit_in_bins, args.min_number_of_bin, args.max_number_of_bin, args.min_pixel_displacment);
	focal_length_calibrator.setIntrinsicMatrix(K_init);
	PrincipalCalibrator principalpoint_calibrator(args.undistorted_ROI, args.feature_limit_in_bins, args.min_number_of_bin, args.max_number_of_bin, args.min_pixel_displacment,10);
	principalpoint_calibrator.setIntrinsicMatrix(K_init);
	RadialDistortionCalibrator distortion_calibrator(args.undistorted_ROI, args.feature_limit_for_distortion_calculation, args.min_number_of_bin, args.max_number_of_bin, 3);

	size_t step_size_local = args.step_size;


	for (size_t index = args.starting_frame; index < args.parsed_frame_number-5; index += step_size_local)
	{
		std::cout << "Step : " << index << " with a stepsize : " << step_size_local << std::endl;

		Eigen::Matrix4d M1 = detector.calculateRelativeOdometry(sequence.getPoses(), index, step_size_local);
		/*if (!detector.checkOdodmetry(M1, args.minimum_speed, args.max_yaw_limit, true, true))
		{
			std::cout << "ODOMETRY is not accurate enough!" << std::endl;
			continue;
		}*/
		auto image1 = sequence.getImageById(index);
		auto image2 = sequence.getImageById(index + step_size_local);

		const auto matched_featurepairs = detector.detectAndSortFeatures(image1, image2, M1, index, step_size_local);

		distortion_calibrator.addFeatures(matched_featurepairs);
		focal_length_calibrator.addFeatures(matched_featurepairs);
		principalpoint_calibrator.addFeatures(matched_featurepairs);
	}

	//TODO: Redesign the calibrator class.
	int iteration = 0;
	while (iteration < 5)
	{
		focal_length_calibrator.setIntrinsicMatrix(principalpoint_calibrator.getIntrinsicMatrix());

		focal_length_calibrator.calibrate();

		principalpoint_calibrator.setIntrinsicMatrix(focal_length_calibrator.getIntrinsicMatrix());

		principalpoint_calibrator.calibrate();
		iteration++;
	}
	std::cout << "The result of the iterative calibration is: " << std::endl;
	std::cout << principalpoint_calibrator.getIntrinsicMatrix() << std::endl;


	distortion_calibrator.setIntrinsicMatrix(principalpoint_calibrator.getIntrinsicMatrix());
	distortion_calibrator.calibrateWithOdometry();

	//focal_length_calibrator.exportShapeOfProblemSpace("05", "150");
	//principalpoint_calibrator.exportShapeOfProblemSpace("05", "150");
}