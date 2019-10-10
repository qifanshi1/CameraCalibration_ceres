#pragma once
#include<vector>
#include<Eigen/Dense>

double computeError(const Eigen::Vector2d& object_point,
	const Eigen::Vector2d& image_point,
	const Eigen::Matrix3d& K,
	const Eigen::Matrix3d& R,
	const Eigen::Vector3d& T)
{
	Eigen::Vector3d object_point_homo;
	object_point_homo << object_point(0, 0), object_point(1, 0), 0;
	Eigen::Vector3d object_point_camera = R * object_point_homo + T;
	double xc = object_point_camera(0, 0) / object_point_camera(2, 0);
	double yc = object_point_camera(1, 0) / object_point_camera(2, 0);

	double u = xc * K(0, 0) + K(0, 2);
	double v = yc * K(1, 1) + K(1, 2);
	Eigen::Vector2d error_vec;
	error_vec << u - image_point(0, 0), v - image_point(1, 0);

	return error_vec.norm();
}