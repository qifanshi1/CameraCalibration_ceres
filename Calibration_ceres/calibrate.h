#pragma once
#include<opencv2/opencv.hpp>
//Eigen library should be in the forward
#include<Eigen/Dense>
#include<Eigen/Sparse>
#include<opencv2/core/eigen.hpp>

void findHomoList(const std::vector<std::vector<Eigen::Vector2d>>& object_points,
	const std::vector<std::vector<Eigen::Vector2d>>& image_points,
	std::vector<Eigen::Matrix3d>& homo_list)
{
	std::vector<std::vector<cv::Point2d>> object_points_vec;
	std::vector<std::vector<cv::Point2d>> image_points_vec;

	object_points_vec.resize(object_points.size());
	image_points_vec.resize(image_points.size());

	for (int i = 0; i < object_points.size(); i++)
		for (int j = 0; j < object_points[i].size(); j++)
		{
			cv::Point2d pt(object_points[i][j](0, 0), object_points[i][j](1, 0));
			object_points_vec[i].push_back(pt);
			pt.x = image_points[i][j](0, 0);
			pt.y = image_points[i][j](1, 0);
			image_points_vec[i].push_back(pt);
		}


	for (int i = 0; i < object_points.size(); i++)
	{
		cv::Mat homography = cv::findHomography(object_points_vec[i], image_points_vec[i], cv::RANSAC);
		Eigen::Matrix<double, 3, 3> homo_eigen;
		cv::cv2eigen(homography, homo_eigen);
		homo_list.push_back(homo_eigen);
	}

	std::cout << "homography size : " << homo_list.size() << std::endl;

}

Eigen::Matrix<double, 2, 6> constructVectorV(const Eigen::Matrix3d& homography)
{
	Eigen::Matrix<double, 2, 6> v;
	double h11, h12, h13, h21, h22, h23;
	h11 = homography(0, 0);
	h12 = homography(1, 0);
	h13 = homography(2, 0);
	h21 = homography(0, 1);
	h22 = homography(1, 1);
	h23 = homography(2, 1);
	v << h11 * h21, h11*h22 + h12 * h21, h12*h22, h13*h21 + h11 * h23, h13*h22 + h12 * h23, h13*h23,
		(h11*h11 - h21 * h21), (h11*h12 + h12 * h11 - h21 * h22 - h22 * h21), (h12*h12 - h22 * h22),
		(h13*h11 + h11 * h13 - h23 * h21 - h21 * h23), (h13*h12 + h12 * h13 - h23 * h22 - h22 * h23), (h13*h13 - h23 * h23);
	return v;
}

void findInitialK(const std::vector<Eigen::Matrix3d>& homo_list,
	Eigen::Matrix3d& InitialK)
{
	Eigen::Matrix<double, 10, 6> vector_v;
	for (int i = 0; i < homo_list.size(); i++)
	{
		vector_v.block<2, 6>(2 * i, 0) = constructVectorV(homo_list[i]);
	}

	//solve AX=0, A=UDVt, x is the last column of the v,or x is the min eigenvalue of the AAt 
	Eigen::JacobiSVD<Eigen::Matrix<double, 10, 6>> svd(vector_v, Eigen::ComputeFullV);
	Eigen::Matrix<double, 6, 1> b = svd.matrixV().col(5);

	//get the K from the matrix b
	double v0 = (b(1, 0)*b(3, 0) - b(0, 0)*b(4, 0)) / (b(0, 0)*b(2, 0) - b(1, 0)*b(1, 0));
	double lamda = b(5, 0) - (b(3, 0)*b(3, 0) + v0 * (b(1, 0)*b(3, 0) - b(0, 0)*b(4, 0))) / b(0, 0);
	double alpha = std::sqrt(lamda / b(0, 0));
	double beta = std::sqrt(lamda*b(0, 0) / (b(0, 0)*b(2, 0) - b(1, 0)*b(1, 0)));
	double gama = -b(1, 0)*alpha*alpha*beta / lamda;
	double u0 = lamda * v0 / beta - b(3, 0)*alpha*alpha / lamda;

	InitialK << alpha, gama, u0,
		0, beta, v0,
		0, 0, 1;

	std::cout << " Initial K : " << InitialK << std::endl;
}

void findInitialRT(const std::vector<Eigen::Matrix3d>& homo_list,
	const Eigen::Matrix3d& InitialK,
	std::vector<Eigen::Matrix3d>& R_list,
	std::vector<Eigen::Vector3d>& T_list)
{

	for (int i = 0; i < homo_list.size(); i++)
	{
		Eigen::Vector3d lamda_vec = InitialK.inverse()*homo_list[i].col(0);
		double lamda = 1 / lamda_vec.norm();
		Eigen::Vector3d r1 = lamda * lamda_vec;
		Eigen::Vector3d r2 = lamda * InitialK.inverse()*homo_list[i].col(1);
		Eigen::Vector3d r3 = r1.cross(r2);
		Eigen::Vector3d t = lamda * InitialK.inverse()*homo_list[i].col(2);
		T_list.push_back(t);
		Eigen::Matrix3d R;
		R << r1, r2, r3;
		/*std::cout << "r1 :" << r1 << std::endl;
		std::cout << "r2 :" << r2 << std::endl;
		std::cout << "r3 :" << r3 << std::endl;
		std::cout << "R :" << R << std::endl;*/
		Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
		R = svd.matrixU()*(svd.matrixV()).transpose();
		R_list.push_back(R);
	}
	/*std::cout << " R_list size : " << R_list.size() << std::endl;
	std::cout << " T_list size : " << T_list.size() << std::endl;*/
}

void findDistort(const std::vector<std::vector<Eigen::Vector2d>>& object_points,
	const std::vector<std::vector<Eigen::Vector2d>>& image_points,
	const Eigen::Matrix3d& K,
	const std::vector<Eigen::Matrix3d>& R_list,
	const std::vector<Eigen::Vector3d>& T_list,
	Eigen::Matrix<double, 5, 1>& distort_coeffs)
{
	/*Eigen::Matrix<double, 990, 5> D_matrix;
	Eigen::Matrix<double, 990, 1> d_vector;*/
	Eigen::Matrix<double, 990, 4> D_matrix;
	Eigen::Matrix<double, 990, 1> d_vector;
	double fx = K(0, 0);
	double fy = K(1, 1);
	double cx = K(0, 2);
	double cy = K(1, 2);

	for (int i = 0; i < object_points.size(); i++)
	{
		for (int j = 0; j < object_points[i].size(); j++)
		{
			Eigen::Vector3d pt;
			pt << object_points[i][j](0, 0), object_points[i][j](1, 0), 0;
			Eigen::Vector3d camera_pt = R_list[i] * pt + T_list[i];
			Eigen::Vector2d ideal_camera;
			double x, y;
			x = camera_pt(0, 0) / camera_pt(2, 0);
			y = camera_pt(1, 0) / camera_pt(2, 0);
			ideal_camera << x, y;
			double r = ideal_camera.squaredNorm();

			Eigen::Vector2d distort_camera;
			distort_camera << (image_points[i][j](0, 0) - cx) / fx,
				(image_points[i][j](1, 0) - cy) / fy;

			d_vector.block<2, 1>(i * object_points[i].size() * 2 + j * 2, 0) = distort_camera - ideal_camera;
			/*D_matrix.block<2, 5>(i * object_points[i].size() * 2 + j * 2, 0) << x * r, x*r*r, x*r*r*r, 2 * x*y, r + 2 * x*x,
				y*r, y*r*r, y*r*r*r, r + 2 * y*y, 2 * x*y;*/
			D_matrix.block<2, 4>(i * object_points[i].size() * 2 + j * 2, 0) << x * r, x*r*r, 2 * x*y, r + 2 * x*x,
				y*r, y*r*r, r + 2 * y*y, 2 * x*y;
		}
	}

	/*distort_coeffs = D_matrix.colPivHouseholderQr().solve(d_vector);*/
	Eigen::Vector4d dis_temp = D_matrix.colPivHouseholderQr().solve(d_vector);
	distort_coeffs << dis_temp(0, 0), dis_temp(1, 0), 0, dis_temp(2, 0), dis_temp(3, 0);

	std::cout << "distort_coeffs : " << distort_coeffs << std::endl;
}