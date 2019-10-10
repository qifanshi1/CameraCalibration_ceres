#pragma once
#include<vector>
#include<Eigen/Dense>
#include<ceres/ceres.h>
#include<ceres/rotation.h>


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

double computeReprojectError(const Eigen::Vector2d& object_point,
	const Eigen::Vector2d& image_point,
	const Eigen::Matrix3d& K,
	const Eigen::Matrix3d& R,
	const Eigen::Vector3d& T,
	const Eigen::Matrix<double, 5, 1>& D)
{
	Eigen::Vector3d object_point_homo;
	object_point_homo << object_point(0, 0), object_point(1, 0), 0;
	Eigen::Vector3d object_point_camera = R * object_point_homo + T;
	double xc = object_point_camera(0, 0) / object_point_camera(2, 0);
	double yc = object_point_camera(1, 0) / object_point_camera(2, 0);
	double dist_r = xc * xc + yc * yc;

	double xc_dist = xc * (1 + D(0, 0)*dist_r + D(1, 0)*dist_r*dist_r + D(2, 0)*dist_r*dist_r*dist_r) + 2 * D(3, 0)*xc* yc + D(4, 0)*(dist_r + 2 * xc * xc);
	double yc_dist = yc * (1 + D(0, 0)*dist_r + D(1, 0)*dist_r*dist_r + D(2, 0)*dist_r*dist_r*dist_r) + D(3, 0)*(dist_r + 2 * yc * yc) + 2 * D(4, 0)*xc* yc;

	double u = xc_dist * K(0, 0) + K(0, 2);
	double v = yc_dist * K(1, 1) + K(1, 2);
	Eigen::Vector2d error_vec;
	error_vec << u - image_point(0, 0), v - image_point(1, 0);

	return error_vec.norm();
}

//cost function
struct KRT_COST
{
	KRT_COST(Eigen::Vector2d uv, Eigen::Vector2d xyz) :
		uv_(uv), xyz_(xyz) {}

	template<typename T>
	bool operator()
		(
			const T* const K,
			const T* const R,
			const T* const t,
			T* residual
			)const
	{
		T p[3];
		p[0] = T(xyz_(0, 0));
		p[1] = T(xyz_(1, 0));
		p[2] = T(0);

		T pt_camera[3];
		ceres::AngleAxisRotatePoint(R, p, pt_camera);
		pt_camera[0] += t[0];
		pt_camera[1] += t[1];
		pt_camera[2] += t[2];

		T xp = pt_camera[0] / pt_camera[2];
		T yp = pt_camera[1] / pt_camera[2];

		T u_ = xp * K[0] + K[2];
		T v_ = yp * K[1] + K[3];
		residual[0] = T(uv_[0]) - u_;
		residual[1] = T(uv_[1]) - v_;

		return true;
	}

	const Eigen::Vector2d uv_, xyz_;
};

struct KRDT_COST
{
	KRDT_COST(Eigen::Vector2d uv, Eigen::Vector2d xyz) :
		uv_(uv), xyz_(xyz) {}

	template<typename T>
	bool operator()
		(
			const T* const K,
			const T* const R,
			const T* const t,
			const T* const D,
			T* residual
			)const
	{
		T p[3];
		p[0] = T(xyz_(0, 0));
		p[1] = T(xyz_(1, 0));
		p[2] = T(0);

		T pt_camera[3];
		ceres::AngleAxisRotatePoint(R, p, pt_camera);
		pt_camera[0] += t[0];
		pt_camera[1] += t[1];
		pt_camera[2] += t[2];

		T xp = pt_camera[0] / pt_camera[2];
		T yp = pt_camera[1] / pt_camera[2];


		T dis_r = xp * xp + yp * yp;
		//T temp = T(1) + D[0] * dis_r + D[1] * dis_r*dis_r + D[2] * dis_r*dis_r*dis_r;
		T temp = T(1) + D[0] * dis_r + D[1] * dis_r*dis_r;

		T xp_d = xp * temp + T(2) * D[3] * xp *yp + D[4] * (dis_r + T(2) * xp * xp);
		T yp_d = yp * temp + D[3] * (dis_r + T(2) * yp * yp) + T(2) * D[4] * xp *yp;

		T u_ = xp_d * K[0] + K[2];
		T v_ = yp_d * K[1] + K[3];
		residual[0] = T(uv_[0]) - u_;
		residual[1] = T(uv_[1]) - v_;

		return true;

	}

	const Eigen::Vector2d uv_, xyz_;
};

void optimizeKRT(const std::vector<std::vector<Eigen::Vector2d>>& object_points,
	const std::vector<std::vector<Eigen::Vector2d>>& image_points,
	Eigen::Matrix3d& InitialK,
	std::vector<Eigen::Matrix3d>& R_list,
	std::vector<Eigen::Vector3d>& T_list)
{
	//k[4] is fx,fy,cx,cy
	double k[4] = { InitialK(0,0),InitialK(1,1),InitialK(0,2),InitialK(1,2) };
	double r[5][3], t[5][3];
	for (int i = 0; i < object_points.size(); i++)
	{
		Eigen::AngleAxisd R_angle(R_list[i]);
		Eigen::Vector3d R_vec = R_angle.angle()*R_angle.axis();
		for (int j = 0; j < 3; j++)
		{
			r[i][j] = R_vec(j, 0);
			t[i][j] = T_list[i](j, 0);
		}
	}

	ceres::Problem problem;
	for (int i = 0; i < object_points.size(); i++)
	{
		for (int j = 0; j < object_points[i].size(); j++)
		{
			problem.AddResidualBlock(
				new ceres::AutoDiffCostFunction<KRT_COST, 2, 4, 3, 3>(
					new KRT_COST(image_points[i][j], object_points[i][j])
					),
				nullptr,
				k,
				r[i],
				t[i]
			);
		}
	}
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	for (int i = 0; i < object_points.size(); i++)
	{
		//update the R,T
		Eigen::Vector3d R_vec_opt(r[i][0], r[i][1], r[i][2]);
		Eigen::AngleAxisd R_angle_opt(R_vec_opt.norm(), R_vec_opt / R_vec_opt.norm());
		Eigen::Matrix3d R_opt(R_angle_opt);
		R_list[i] = R_opt;
		Eigen::Vector3d t_vec_opt(t[i][0], t[i][1], t[i][2]);
		T_list[i] = t_vec_opt;
		/*std::cout << " R_opt : " << R_opt << std::endl;
		std::cout << "t_vec_opt : " << t_vec_opt << std::endl;
		std::cout << "k : " << k[0] << "  " << k[1] << "  " << k[2] << "  " << k[3] << std::endl;*/
	}

	InitialK << k[0], 0, k[2],
		0, k[1], k[3],
		0, 0, 1;
	std::cout << "optimized_K : " << InitialK << std::endl;
}

void optimizeKDRT(const std::vector<std::vector<Eigen::Vector2d>>& object_points,
	const std::vector<std::vector<Eigen::Vector2d>>& image_points,
	Eigen::Matrix3d& InitialK,
	std::vector<Eigen::Matrix3d>& R_list,
	std::vector<Eigen::Vector3d>& T_list,
	Eigen::Matrix<double, 5, 1>& distort_coeffs
)
{
	//k[4] is fx,fy,cx,cy
	double k[4] = { InitialK(0,0),InitialK(1,1),InitialK(0,2),InitialK(1,2) };
	//double d[5] = { distort_coeffs(0,0),distort_coeffs(1,0),distort_coeffs(2,0), distort_coeffs(3,0), distort_coeffs(4,0) };
	double d[5] = { distort_coeffs(0,0),distort_coeffs(1,0),0, distort_coeffs(3,0), distort_coeffs(4,0) };
	double r[5][3], t[5][3];
	for (int i = 0; i < object_points.size(); i++)
	{
		Eigen::AngleAxisd R_angle(R_list[i]);
		Eigen::Vector3d R_vec = R_angle.angle()*R_angle.axis();
		for (int j = 0; j < 3; j++)
		{
			r[i][j] = R_vec(j, 0);
			t[i][j] = T_list[i](j, 0);
		}
	}

	std::vector<std::vector<Eigen::Vector2d>> object_points_current(object_points.begin(), object_points.end());
	std::vector<std::vector<Eigen::Vector2d>> image_points_current(image_points.begin(), image_points.end());

	std::vector<std::vector<Eigen::Vector2d>> object_points_temp, image_points_temp;
	object_points_temp.resize(object_points.size());
	image_points_temp.resize(image_points.size());
	int iter = 0;
	bool flag = true;

	while (flag && (iter < 3))
	{
		flag = false;
		ceres::Problem problem;
		for (int i = 0; i < object_points_current.size(); i++)
		{
			for (int j = 0; j < object_points_current[i].size(); j++)
			{
				problem.AddResidualBlock(
					new ceres::AutoDiffCostFunction<KRDT_COST, 2, 4, 3, 3, 5>(
						new KRDT_COST(image_points_current[i][j], object_points_current[i][j])
						),
					nullptr,
					k,
					r[i],
					t[i],
					d
				);
			}
		}
		ceres::Solver::Options options;
		options.linear_solver_type = ceres::DENSE_SCHUR;
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);

		InitialK << k[0], 0, k[2],
			0, k[1], k[3],
			0, 0, 1;
		std::cout << "optimized_K : " << InitialK << std::endl;
		distort_coeffs << d[0], d[1], d[2], d[3], d[4];
		std::cout << "distort_coeffs : " << distort_coeffs << std::endl;

		for (int i = 0; i < object_points_current.size(); i++)
		{
			//update the R,T
			Eigen::Vector3d R_vec_opt(r[i][0], r[i][1], r[i][2]);
			Eigen::AngleAxisd R_angle_opt(R_vec_opt.norm(), R_vec_opt / R_vec_opt.norm());
			Eigen::Matrix3d R_opt(R_angle_opt);
			R_list[i] = R_opt;
			Eigen::Vector3d t_vec_opt(t[i][0], t[i][1], t[i][2]);
			T_list[i] = t_vec_opt;
			/*std::cout << " R_opt : " << R_opt << std::endl;
			std::cout << "t_vec_opt : " << t_vec_opt << std::endl;
			std::cout << "k : " << k[0] << "  " << k[1] << "  " << k[2] << "  " << k[3] << std::endl;*/
		}
		double error = 0.0;
		for (int i = 0; i < object_points_current.size(); i++)
		{
			for (int j = 0; j < object_points_current[i].size(); j++)
			{
				error = computeReprojectError(object_points_current[i][j], image_points_current[i][j],
					InitialK, R_list[i], T_list[i], distort_coeffs);
				if (error > 0.06)
					flag = true;
				else
				{
					object_points_temp[i].push_back(object_points_current[i][j]);
					image_points_temp[i].push_back(image_points_current[i][j]);
				}

			}
		}
		bool bifinished = false;
		for (auto pt : object_points_temp)
		{
			if (pt.size() < 10)
				bifinished = true;
		}
		if (bifinished)
			break;

		object_points_current.assign(object_points_temp.begin(), object_points_temp.end());
		image_points_current.assign(image_points_temp.begin(), image_points_temp.end());
		iter++;
	}

	double error = 0.0;
	for (int i = 0; i < object_points_current.size(); i++)
	{
		double error_lone = 0.0;
		for (int j = 0; j < object_points_current[i].size(); j++)
		{
			double error_temp = computeReprojectError(object_points_current[i][j], image_points_current[i][j],
				InitialK, R_list[i], T_list[i], distort_coeffs);
			error_lone += error_temp;
			error += error_temp;
		}
		std::cout << " error_lone : " << error_lone / object_points[i].size() << std::endl;
	}
	std::cout << " error : " << error / (object_points[0].size()* object_points.size()) << std::endl;

}


