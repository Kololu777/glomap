// rigid3d_torch_utils.h
#pragma once

#include "glomap/math/rigid3d.h"
#include "glomap/math/rigid3d_torch.h"
#include <torch/torch.h>

namespace glomap {

    // Conversion tolerance for floating point comparisons
    constexpr double kConversionTolerance = 1e-10;

    // Conversion functions between Eigen and LibTorch
    class RigidConverter {
    public:
        static Rigid3dTorch EigenToTorch(const Rigid3d& eigen_transform) {
            // Convert quaternion (Eigen stores as [x,y,z,w], LibTorch as [w,x,y,z])
            torch::Tensor rotation = torch::tensor({
                                                       eigen_transform.rotation.w(),
                                                       eigen_transform.rotation.x(),
                                                       eigen_transform.rotation.y(),
                                                       eigen_transform.rotation.z()
                                                   }, torch::kFloat64);

            // Convert translation
            torch::Tensor translation = torch::tensor({
                                                          eigen_transform.translation.x(),
                                                          eigen_transform.translation.y(),
                                                          eigen_transform.translation.z()
                                                      }, torch::kFloat64);

            return Rigid3dTorch(rotation, translation);
        }

        static Rigid3d TorchToEigen(const Rigid3dTorch& torch_transform) {
            const auto& q = torch_transform.quaternion();
            const auto& t = torch_transform.translation();

            Eigen::Quaterniond rotation(
                q[0].item<double>(), // w
                q[1].item<double>(), // x
                q[2].item<double>(), // y
                q[3].item<double>()  // z
            );

            Eigen::Vector3d translation(
                t[0].item<double>(),
                t[1].item<double>(),
                t[2].item<double>()
            );

            return Rigid3d(rotation, translation);
        }
    };

    // Equality testing utilities
    class RigidComparator {
    public:
        // Check if two transforms are approximately equal
        static bool IsApproxEqual(const Rigid3d& eigen_transform,
                                  const Rigid3dTorch& torch_transform,
                                  double tolerance = kConversionTolerance) {
            // Convert torch to eigen for comparison
            Rigid3d converted = RigidConverter::TorchToEigen(torch_transform);

            // Compare quaternions - handle double cover of quaternions
            bool quaternion_equal = IsQuaternionApproxEqual(
                eigen_transform.rotation,
                converted.rotation,
                tolerance);

            // Compare translations
            bool translation_equal = IsVectorApproxEqual(
                eigen_transform.translation,
                converted.translation,
                tolerance);

            return quaternion_equal && translation_equal;
        }

        // Test if two transforms produce approximately equal results
        static bool HasApproxEqualEffect(const Rigid3d& eigen_transform,
                                         const Rigid3dTorch& torch_transform,
                                         double tolerance = kConversionTolerance) {
            // Test points at different positions
            std::vector<Eigen::Vector3d> test_points = {
                Eigen::Vector3d(1, 0, 0),
                Eigen::Vector3d(0, 1, 0),
                Eigen::Vector3d(0, 0, 1),
                Eigen::Vector3d(1, 1, 1)
            };

            for (const auto& point : test_points) {
                // Transform point with Eigen
                Eigen::Vector3d eigen_result = eigen_transform * point;

                // Transform point with LibTorch
                torch::Tensor torch_point = torch::tensor({
                                                              point.x(), point.y(), point.z()
                                                          }, torch::kFloat64);
                torch::Tensor torch_result = transform_point(torch_transform, torch_point);

                // Compare results
                Eigen::Vector3d converted_result(
                    torch_result[0].item<double>(),
                    torch_result[1].item<double>(),
                    torch_result[2].item<double>()
                );

                if (!IsVectorApproxEqual(eigen_result, converted_result, tolerance)) {
                    return false;
                }
            }
            return true;
        }

    private:
        static bool IsQuaternionApproxEqual(const Eigen::Quaterniond& q1,
                                            const Eigen::Quaterniond& q2,
                                            double tolerance) {
            // Handle double cover of quaternions (q and -q represent same rotation)
            double dot_product = std::abs(
                q1.w() * q2.w() +
                q1.x() * q2.x() +
                q1.y() * q2.y() +
                q1.z() * q2.z()
            );
            return std::abs(dot_product - 1.0) < tolerance;
        }

        static bool IsVectorApproxEqual(const Eigen::Vector3d& v1,
                                        const Eigen::Vector3d& v2,
                                        double tolerance) {
            return (v1 - v2).norm() < tolerance;
        }
    };

} // namespace glomap