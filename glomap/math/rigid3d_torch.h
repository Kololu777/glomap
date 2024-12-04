#pragma once

#include <torch/torch.h>

namespace glomap {

    class Rigid3dTorch {
    public:
        // Default constructor - identity transform
        Rigid3dTorch();

        // Construct from quaternion and translation
        Rigid3dTorch(const torch::Tensor& rotation, const torch::Tensor& translation);

        // Convert rotation to 3x3 matrix
        torch::Tensor rotation_matrix() const;

        // Get translation vector
        const torch::Tensor& translation() const { return translation_; }

        // Get quaternion
        const torch::Tensor& quaternion() const { return rotation_; }

        // Compose two transforms
        Rigid3dTorch operator*(const Rigid3dTorch& other) const;

        // Inverse transform
        Rigid3dTorch inverse() const;

    private:
        // Normalize quaternion to unit length
        static torch::Tensor normalize_quaternion(const torch::Tensor& q);

        // Multiply two quaternions
        static torch::Tensor quaternion_multiply(const torch::Tensor& q1, const torch::Tensor& q2);

        // Internal storage
        torch::Tensor rotation_;    // quaternion [w,x,y,z]
        torch::Tensor translation_; // vector [x,y,z]
    };

    // Utility functions
    torch::Tensor transform_point(const Rigid3dTorch& transform, const torch::Tensor& point);
    double calc_angle(const Rigid3dTorch& pose1, const Rigid3dTorch& pose2);

} // namespace glomap