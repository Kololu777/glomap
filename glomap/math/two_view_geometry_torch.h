#pragma once

#include <torch/torch.h>
#include "glomap/math/rigid3d_torch.h"
#include "glomap/colmap_migration/camera.h"

namespace glomap {

    namespace torch_utils {
        // Utility function to convert point to homogeneous coordinates
        inline torch::Tensor to_homogeneous(const torch::Tensor& x) {
            return torch::cat({x, torch::ones({1}, x.options())});
        }

        // Helper to get double value from tensor
        inline double to_double(const torch::Tensor& t) {
            return t.item().toDouble();
        }
    }

// Main interface functions
    bool CheckCheiralityTorch(const Rigid3dTorch& pose,
                              const torch::Tensor& x1,
                              const torch::Tensor& x2,
                              double min_depth = 0.,
                              double max_depth = 100.);

    torch::Tensor GetOrientationSignumTorch(const torch::Tensor& F,
                                            const torch::Tensor& epipole,
                                            const torch::Tensor& pt1,
                                            const torch::Tensor& pt2);

    void EssentialFromMotionTorch(const Rigid3dTorch& pose, torch::Tensor* E);

    void FundamentalFromMotionAndCamerasTorch(const Camera& camera1,
                                              const Camera& camera2,
                                              const Rigid3dTorch& pose,
                                              torch::Tensor* F);

    double SampsonError2DTorch(const torch::Tensor& E,
                               const torch::Tensor& x1,
                               const torch::Tensor& x2);

    double SampsonError3DTorch(const torch::Tensor& E,
                               const torch::Tensor& x1_3d,
                               const torch::Tensor& x2_3d);

    double HomographyErrorTorch(const torch::Tensor& H,
                                const torch::Tensor& x1,
                                const torch::Tensor& x2);

} // namespace glomap