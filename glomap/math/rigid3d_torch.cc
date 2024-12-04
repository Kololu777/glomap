#include "glomap/math/rigid3d_torch.h"
#include "glomap/colmap_migration/logging.h"
#include <cmath>

namespace glomap {

    Rigid3dTorch::Rigid3dTorch()
        : rotation_(torch::tensor({1.0, 0.0, 0.0, 0.0}, torch::kFloat64)), // w,x,y,z
          translation_(torch::zeros({3}, torch::kFloat64)) {}

    Rigid3dTorch::Rigid3dTorch(const torch::Tensor& rotation, const torch::Tensor& translation)
        : rotation_(normalize_quaternion(rotation.to(torch::kFloat64))),
          translation_(translation.to(torch::kFloat64)) {
        CHECK(rotation_.dim() == 1) << "Rotation must be 1-dimensional";
        CHECK(rotation_.size(0) == 4) << "Rotation must have 4 elements";
        CHECK(translation_.dim() == 1) << "Translation must be 1-dimensional";
        CHECK(translation_.size(0) == 3) << "Translation must have 3 elements";
    }


    torch::Tensor Rigid3dTorch::rotation_matrix() const {
        // q = [w, x, y, z]
        double w = rotation_[0].item<double>();
        double x = rotation_[1].item<double>();
        double y = rotation_[2].item<double>();
        double z = rotation_[3].item<double>();

        return torch::tensor(
            {{1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y},
             {2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x},
             {2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y}},
            torch::kFloat64);
    }

    Rigid3dTorch Rigid3dTorch::operator*(const Rigid3dTorch& other) const {
        torch::Tensor new_rotation = quaternion_multiply(rotation_, other.rotation_);
        torch::Tensor new_translation = translation_ +
                                        torch::matmul(rotation_matrix(), other.translation_);
        return Rigid3dTorch(new_rotation, new_translation);
    }

    Rigid3dTorch Rigid3dTorch::inverse() const {
        // Create a new tensor with negated values for x,y,z components
        auto inv_rotation = torch::stack({
            rotation_[0],
            -rotation_[1],
            -rotation_[2],
            -rotation_[3]
        });

        auto inv_translation = -torch::matmul(
            rotation_matrix().transpose(0, 1),
            translation_);
        return Rigid3dTorch(inv_rotation, inv_translation);
    }

    torch::Tensor Rigid3dTorch::normalize_quaternion(const torch::Tensor& q) {
        return q / torch::norm(q);
    }

    torch::Tensor Rigid3dTorch::quaternion_multiply(const torch::Tensor& q1, const torch::Tensor& q2) {
        double w1 = q1[0].item<double>();
        double x1 = q1[1].item<double>();
        double y1 = q1[2].item<double>();
        double z1 = q1[3].item<double>();

        double w2 = q2[0].item<double>();
        double x2 = q2[1].item<double>();
        double y2 = q2[2].item<double>();
        double z2 = q2[3].item<double>();

        return torch::tensor({
                                 w1*w2 - x1*x2 - y1*y2 - z1*z2,
                                 w1*x2 + x1*w2 + y1*z2 - z1*y2,
                                 w1*y2 - x1*z2 + y1*w2 + z1*x2,
                                 w1*z2 + x1*y2 - y1*x2 + z1*w2
                             }, torch::kFloat64);
    }

    torch::Tensor transform_point(const Rigid3dTorch& transform, const torch::Tensor& point) {
        return torch::matmul(transform.rotation_matrix(), point) + transform.translation();
    }

    double calc_angle(const Rigid3dTorch& pose1, const Rigid3dTorch& pose2) {
        torch::Tensor rot_diff = torch::matmul(
            pose1.rotation_matrix().transpose(0, 1),
            pose2.rotation_matrix()
        );
        double trace = rot_diff.trace().item<double>();
        double cos_angle = (trace - 1.0) / 2.0;
        cos_angle = std::min(std::max(cos_angle, -1.0), 1.0);
        return std::acos(cos_angle) * 180.0 / M_PI;
    }

} // namespace glomap