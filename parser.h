#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <experimental/filesystem>
#include <filesystem>
#include <optional>
#include <map>
#include <Eigen/Dense>




/// Structure for each frame with timestamp, id, and image.
struct Frame
{
	size_t id = 0;
	size_t timestamp = 0;
	std::string path_of_image;
	cv::Mat image;
};
/// Datastructure for pose matricies
struct Pose
{
public:
	size_t id = 0;
	Eigen::Matrix4d pose_of_0camera;
public:
	Pose(size_t id, const Eigen::Matrix4d& matrix) : id(id), pose_of_0camera(matrix)
	{
	}


};

/// ImageSequence class is responsible for loading and storing images from the Kitti dataset
class ImageSequence
{
private:
	
	 std::map<size_t,Frame> image_sequence;
	 std::map<std::string,Eigen::Matrix4d> cameraMatrix;
	 std::vector<Pose>			   poses;
	 cv::Size2i image_size = { 0,0 };
	 bool					does_image_size_set = false;
	 bool					is_projection_matrix;

private:
	/// loadImageSeq is responsible for loading a sequence of images with given stepsize in a given length(how many pictures we want to load)
	/// "path" the path to the current datafolder
	/// "length" tells us how long does the sequence needed to be
	/// "step_size" determines the distance between the loaded frames. // TODO: This is no longer used.
	void loadImageSeq(const std::string& path, size_t length);

	void loadAiDriveImageSeq(const std::string& path, size_t length);

	/// Loads poses with a dataset given at a path "file_name"
	void loadPoses(const std::string& file_name, size_t length);

	void loadAiDrivePoses(const std::string& file_name, size_t length);

	/// Load the camera matricies from the KITTI dataset, boolean projectionMatrix variable is not used yet
	void loadCameraMatrix(const std::string& path_to_calibfile, bool projectionMatrixNeeded);

public:

	ImageSequence(const std::string& path_to_images, const std::string& path_to_poses, const std::string& path_to_calibfile, size_t length);

//	ImageSequence(const std::string& path_to_images, const std::string& path_to_poses, const std::string& path_to_calibfile, size_t length);

	/// Returns the previously loaded image_sequence
	std::map<size_t, Frame> getImageSeq() const; // TODO: Rather return a reference to the data. (This currently makes a deep copy.)

	/// Returns an image by its ID
	// id which frame do you want to acces from the loaded sequence (starts from 0)
	cv::Mat getImageById(size_t id) const; // TODO: Just a sidenote: Here the return type is good, because "cv::Mat" works like a "shared_ptr", in that it only points the contained image's memory...

	/// Returns the parsed poses
	const std::vector<Pose>& getPoses() const;

	const cv::Size getImageSize() const;

	/// Returns the loaded camera matricies.
	Eigen::Matrix4d& getCameraMatrix(const std::string& camera_id);

	Eigen::Matrix3d getIntrinsicMatrix(const std::string& camera_id) const;

	/// If Image sequence and Poses are loaded from the dataset then itt will order the sequence with regards to their ID.
	/// It provides the correctly mathed Frame-Pose paires for later calculation.
	std::vector<std::pair<Frame, Pose>> makeFramePosePairs() const;

};
