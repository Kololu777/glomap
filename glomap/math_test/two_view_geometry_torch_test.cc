#include "glomap/math/two_view_geometry_torch.h"
#include "glomap/math/two_view_geometry.h"
#include "glomap/math/rigid3d_torch_utils.h"
#include <gtest/gtest.h>
#include <random>

namespace glomap {
    namespace {

        // Test fixture with common setup and utility functions
        class TwoViewGeometryTest : public testing::Test {
        protected:
            void SetUp() override {
                SetupBasicTransform();
                SetupTestPoints();
                ComputeEssentialMatrices();
            }

            void SetupBasicTransform() {
                // Create a sample pose with known rotation and translation
                Eigen::Quaterniond rotation(Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitX()));
                Eigen::Vector3d translation(1.0, 0.0, 0.0);
                eigen_pose = Rigid3d(rotation, translation);
                torch_pose = RigidConverter::EigenToTorch(eigen_pose);
            }

            void SetupTestPoints() {
                // Create various test points including edge cases
                x1_eigen = Eigen::Vector3d(0.0, 0.0, 1.0);
                x2_eigen = Eigen::Vector3d(0.1, 0.0, 1.0);
                x1_torch = torch::tensor({0.0, 0.0, 1.0}, torch::kFloat64);
                x2_torch = torch::tensor({0.1, 0.0, 1.0}, torch::kFloat64);

                x1_2d_eigen = x1_eigen.head<2>();
                x2_2d_eigen = x2_eigen.head<2>();
                x1_2d_torch = torch::tensor({0.0, 0.0}, torch::kFloat64);
                x2_2d_torch = torch::tensor({0.1, 0.0}, torch::kFloat64);

                // Edge cases
                point_at_infinity_eigen = Eigen::Vector3d(1.0, 1.0, 0.0);
                point_at_infinity_torch = torch::tensor({1.0, 1.0, 0.0}, torch::kFloat64);

                point_at_origin_eigen = Eigen::Vector3d(0.0, 0.0, 0.0);
                point_at_origin_torch = torch::tensor({0.0, 0.0, 0.0}, torch::kFloat64);
            }

            void ComputeEssentialMatrices() {
                EssentialFromMotion(eigen_pose, &E_eigen);
                EssentialFromMotionTorch(torch_pose, &E_torch);
            }

            static std::pair<Eigen::Vector3d, torch::Tensor> GenerateRandomPoint(double min_z = 0.1) {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<> dis(-1.0, 1.0);
                std::uniform_real_distribution<> dis_z(min_z, 2.0);

                Eigen::Vector3d eigen_pt(dis(gen), dis(gen), dis_z(gen));
                torch::Tensor torch_pt = torch::tensor(
                    {eigen_pt.x(), eigen_pt.y(), eigen_pt.z()},
                    torch::kFloat64);

                return {eigen_pt, torch_pt};
            }

            static Camera CreateTestCamera() {
                Camera camera = Camera::CreateFromModelName(
                    0,                    // camera_id
                    "SIMPLE_PINHOLE",     // model_name
                    1000,                 // focal_length
                    1000,                 // width
                    1000                  // height
                );
                camera.has_prior_focal_length = true;
                return camera;
            }

            static bool IsWithinRelativeTolerance(double val1, double val2, double rel_tol = kRelativeTolerance) {
                double abs_diff = std::abs(val1 - val2);
                double max_abs = std::max(std::abs(val1), std::abs(val2));
                if (max_abs < 1e-10) {  // For values very close to zero
                    return abs_diff < rel_tol;
                }
                return abs_diff / max_abs < rel_tol;
            }

            Rigid3d eigen_pose;
            Rigid3dTorch torch_pose;
            Eigen::Matrix3d E_eigen;
            torch::Tensor E_torch;

            Eigen::Vector3d x1_eigen, x2_eigen;
            Eigen::Vector2d x1_2d_eigen, x2_2d_eigen;
            torch::Tensor x1_torch, x2_torch;
            torch::Tensor x1_2d_torch, x2_2d_torch;

            Eigen::Vector3d point_at_infinity_eigen, point_at_origin_eigen;
            torch::Tensor point_at_infinity_torch, point_at_origin_torch;

            static constexpr double kTestTolerance = 1e-6;
            static constexpr double kRelativeTolerance = 1e-6;
        };

        TEST_F(TwoViewGeometryTest, EssentialMatrixProperties) {
            // Test essential matrix properties
            // 1. det(E) = 0
            EXPECT_NEAR(E_eigen.determinant(), 0.0, kTestTolerance);
            EXPECT_NEAR(torch::det(E_torch).item<double>(), 0.0, kTestTolerance);

            // 2. 2*E*E^T*E - trace(E*E^T)*E = 0 (Huang-Faugeras constraint)
            auto EEt_eigen = E_eigen * E_eigen.transpose();
            auto constraint_eigen = 2 * E_eigen * EEt_eigen - EEt_eigen.trace() * E_eigen;

            auto EEt_torch = torch::matmul(E_torch, E_torch.transpose(0, 1));
            auto constraint_torch = 2 * torch::matmul(E_torch, EEt_torch) -
                                    torch::trace(EEt_torch).item<double>() * E_torch;

            for(int i = 0; i < 3; i++) {
                for(int j = 0; j < 3; j++) {
                    EXPECT_NEAR(constraint_eigen(i,j), 0.0, 1e-8);
                    EXPECT_NEAR(constraint_torch[i][j].item<double>(), 0.0, 1e-8);
                }
            }
        }

        TEST_F(TwoViewGeometryTest, SampsonError2D) {
            // Test normal case
            double error_eigen = SampsonError(E_eigen, x1_2d_eigen, x2_2d_eigen);
            double error_torch = SampsonError2DTorch(E_torch, x1_2d_torch, x2_2d_torch);
            EXPECT_NEAR(error_eigen, error_torch, kTestTolerance);

            // Test multiple random points
            for (int i = 0; i < 100; ++i) {
                auto [p1_eigen, p1_torch] = GenerateRandomPoint();
                auto [p2_eigen, p2_torch] = GenerateRandomPoint();

                Eigen::Vector2d p1_2d_eigen = p1_eigen.head<2>() / p1_eigen.z();
                Eigen::Vector2d p2_2d_eigen = p2_eigen.head<2>() / p2_eigen.z();

                torch::Tensor p1_2d_torch = p1_torch.slice(0, 0, 2) / p1_torch[2];
                torch::Tensor p2_2d_torch = p2_torch.slice(0, 0, 2) / p2_torch[2];

                error_eigen = SampsonError(E_eigen, p1_2d_eigen, p2_2d_eigen);
                error_torch = SampsonError2DTorch(E_torch, p1_2d_torch, p2_2d_torch);

                EXPECT_NEAR(error_eigen, error_torch, kTestTolerance)
                    << "Failed at iteration " << i;
            }
        }

        TEST_F(TwoViewGeometryTest, SampsonError3D) {
            // Test normal case
            double error_eigen = SampsonError(E_eigen, x1_eigen, x2_eigen);
            double error_torch = SampsonError3DTorch(E_torch, x1_torch, x2_torch);

            EXPECT_TRUE(IsWithinRelativeTolerance(error_eigen, error_torch))
                << "Base case failed"
                << "\nerror_eigen = " << error_eigen
                << "\nerror_torch = " << error_torch;

            // Test multiple random points
            for (int i = 0; i < 100; ++i) {
                auto [p1_eigen, p1_torch] = GenerateRandomPoint();
                auto [p2_eigen, p2_torch] = GenerateRandomPoint();

                error_eigen = SampsonError(E_eigen, p1_eigen, p2_eigen);
                error_torch = SampsonError3DTorch(E_torch, p1_torch, p2_torch);

                EXPECT_NEAR(error_eigen, error_torch, kTestTolerance)
                    << "Failed at iteration " << i;
            }
        }

        TEST_F(TwoViewGeometryTest, CheckCheirality) {
            // Test normal case
            bool result_eigen = CheckCheirality(eigen_pose, x1_eigen, x2_eigen);
            bool result_torch = CheckCheiralityTorch(torch_pose, x1_torch, x2_torch);
            EXPECT_EQ(result_eigen, result_torch);

            // Test points at different depths
            std::vector<double> test_depths = {0.1, 1.0, 10.0, 100.0};
            for (double depth : test_depths) {
                auto [p1_eigen, p1_torch] = GenerateRandomPoint(depth);
                auto [p2_eigen, p2_torch] = GenerateRandomPoint(depth);

                result_eigen = CheckCheirality(eigen_pose, p1_eigen, p2_eigen);
                result_torch = CheckCheiralityTorch(torch_pose, p1_torch, p2_torch);
                EXPECT_EQ(result_eigen, result_torch)
                    << "Failed at depth " << depth;
            }
        }

        TEST_F(TwoViewGeometryTest, FundamentalMatrix) {
            Camera camera1 = CreateTestCamera();
            Camera camera2 = CreateTestCamera();

            // Get essential matrices
            Eigen::Matrix3d E_eigen;
            torch::Tensor E_torch;

            EssentialFromMotion(eigen_pose, &E_eigen);
            EssentialFromMotionTorch(torch_pose, &E_torch);

            // Compare fundamental matrix computation steps
            Eigen::Matrix3d F_eigen;
            torch::Tensor F_torch;

            // Step-by-step Eigen computation for comparison
            Eigen::Matrix3d K1 = camera1.GetK();
            Eigen::Matrix3d K2 = camera2.GetK();

            Eigen::Matrix3d K2_inv_t = K2.transpose().inverse();
            Eigen::Matrix3d K1_inv = K1.inverse();

            Eigen::Matrix3d step1_eigen = K2_inv_t * E_eigen;

            Eigen::Matrix3d step2_eigen = step1_eigen * K1_inv;

            F_eigen = step2_eigen;
            F_eigen.normalize();

            FundamentalFromMotionAndCamerasTorch(camera1, camera2, torch_pose, &F_torch);
            // Compare normalized results
            SCOPED_TRACE("Fundamental Matrix Comparison");
            for(int i = 0; i < 3; i++) {
                for(int j = 0; j < 3; j++) {
                    EXPECT_NEAR(F_eigen(i,j), F_torch[i][j].item<double>(), kTestTolerance)
                        << "Fundamental matrix differs at (" << i << "," << j << ")";
                }
            }
        }

        TEST_F(TwoViewGeometryTest, HomographyError) {
            // Create a test homography matrix
            Eigen::Matrix3d H_eigen = Eigen::Matrix3d::Identity();
            H_eigen(0,2) = 0.1;  // Add some translation
            torch::Tensor H_torch = torch::eye(3, torch::kFloat64);
            H_torch[0][2] = 0.1;

            // Test normal case
            double error_eigen = HomographyError(H_eigen, x1_2d_eigen, x2_2d_eigen);
            double error_torch = HomographyErrorTorch(H_torch, x1_2d_torch, x2_2d_torch);
            EXPECT_NEAR(error_eigen, error_torch, kTestTolerance);

            // Test multiple random points
            for (int i = 0; i < 100; ++i) {
                auto [p1_eigen, p1_torch] = GenerateRandomPoint();
                auto [p2_eigen, p2_torch] = GenerateRandomPoint();

                Eigen::Vector2d p1_2d_eigen = p1_eigen.head<2>();
                Eigen::Vector2d p2_2d_eigen = p2_eigen.head<2>();

                torch::Tensor p1_2d_torch = p1_torch.slice(0, 0, 2);
                torch::Tensor p2_2d_torch = p2_torch.slice(0, 0, 2);

                error_eigen = HomographyError(H_eigen, p1_2d_eigen, p2_2d_eigen);
                error_torch = HomographyErrorTorch(H_torch, p1_2d_torch, p2_2d_torch);

                EXPECT_NEAR(error_eigen, error_torch, kTestTolerance)
                    << "Failed at iteration " << i;
            }
        }

        TEST_F(TwoViewGeometryTest, OrientationSignum) {
            // Create test fundamental matrix and epipole
            Eigen::Matrix3d F_eigen;
            F_eigen << 0, -1, 0.2,
                1,  0, -0.1,
                -0.2, 0.1, 0;

            torch::Tensor F_torch = torch::zeros({3, 3}, torch::kFloat64);
            F_torch[0][1] = -1;
            F_torch[0][2] = 0.2;
            F_torch[1][0] = 1;
            F_torch[1][2] = -0.1;
            F_torch[2][0] = -0.2;
            F_torch[2][1] = 0.1;

            // Create test epipole (can be found as the null space of F)
            Eigen::Vector3d epipole_eigen(0.1, 0.2, 1.0);
            torch::Tensor epipole_torch = torch::tensor({0.1, 0.2, 1.0}, torch::kFloat64);

            // Test cases:
            struct TestPoint {
                Eigen::Vector2d pt_eigen;
                torch::Tensor pt_torch;
                TestPoint(double x, double y) :
                                                pt_eigen(x, y),
                                                pt_torch(torch::tensor({x, y}, torch::kFloat64)) {}
            };

            std::vector<std::pair<TestPoint, TestPoint>> test_points = {
                // Normal case
                {TestPoint(0.0, 0.0), TestPoint(0.1, 0.1)},
                // Points around epipole
                {TestPoint(0.1, 0.2), TestPoint(0.15, 0.25)},
                // Symmetric points
                {TestPoint(-1.0, -1.0), TestPoint(1.0, 1.0)},
                // Points far from epipole
                {TestPoint(10.0, 10.0), TestPoint(-10.0, -10.0)},
                // Points with one zero coordinate
                {TestPoint(0.0, 1.0), TestPoint(1.0, 0.0)}
            };

            // Test each pair of points
            for (size_t i = 0; i < test_points.size(); ++i) {
                const auto& [pt1, pt2] = test_points[i];

                double signum_eigen = GetOrientationSignum(F_eigen, epipole_eigen,
                                                           pt1.pt_eigen, pt2.pt_eigen);
                double signum_torch = GetOrientationSignumTorch(F_torch, epipole_torch,
                                                                pt1.pt_torch, pt2.pt_torch).item<double>();

                EXPECT_NEAR(signum_eigen, signum_torch, kTestTolerance)
                    << "Failed for test points at index " << i;

                // Test sign consistency
                EXPECT_EQ(std::signbit(signum_eigen), std::signbit(signum_torch))
                    << "Sign mismatch for test points at index " << i;
            }

            // Test random points
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(-10.0, 10.0);

            for (int i = 0; i < 100; ++i) {
                TestPoint pt1(dis(gen), dis(gen));
                TestPoint pt2(dis(gen), dis(gen));

                double signum_eigen = GetOrientationSignum(F_eigen, epipole_eigen,
                                                           pt1.pt_eigen, pt2.pt_eigen);
                double signum_torch = GetOrientationSignumTorch(F_torch, epipole_torch,
                                                                pt1.pt_torch, pt2.pt_torch).item<double>();

                EXPECT_TRUE(IsWithinRelativeTolerance(signum_eigen, signum_torch))
                    << "Failed for random points iteration " << i
                    << "\nsignum_eigen = " << signum_eigen
                    << "\nsignum_torch = " << signum_torch;

                // Test sign consistency (which is more important than exact values)
                EXPECT_EQ(std::signbit(signum_eigen), std::signbit(signum_torch))
                    << "Sign mismatch for random points iteration " << i;
            }

            // For values near zero, use absolute tolerance
            TestPoint pt_near_zero1(0.0001, 0.0001);
            TestPoint pt_near_zero2(0.0002, 0.0002);

            double signum_eigen = GetOrientationSignum(F_eigen, epipole_eigen,
                                                       pt_near_zero1.pt_eigen, pt_near_zero2.pt_eigen);
            double signum_torch = GetOrientationSignumTorch(F_torch, epipole_torch,
                                                            pt_near_zero1.pt_torch, pt_near_zero2.pt_torch).item<double>();

            EXPECT_NEAR(signum_eigen, signum_torch, kTestTolerance)
                << "Failed for near-zero case";
        }

    } // namespace
} // namespace glomap