#include "glomap/math/two_view_geometry_torch.h"
#include "glomap/colmap_migration/logging.h"
#include "glomap/types.h"

namespace glomap {

    bool CheckCheiralityTorch(const Rigid3dTorch& pose,
                              const torch::Tensor& x1,
                              const torch::Tensor& x2,
                              double min_depth,
                              double max_depth) {
        // This code assumes x1 and x2 are unit vectors
        const torch::Tensor Rx1 = torch::matmul(pose.rotation_matrix(), x1);

        // Compute a and b for [1 a; a 1] * [lambda1; lambda2] = [b1; b2]
        const double a = -torch::dot(Rx1, x2).item<double>();
        const double b1 = -torch::dot(Rx1, pose.translation()).item<double>();
        const double b2 = torch::dot(x2, pose.translation()).item<double>();

        // Drop the factor 1.0/(1-a*a) since it's always positive
        const double lambda1 = b1 - a * b2;
        const double lambda2 = -a * b1 + b2;

        min_depth = min_depth * (1 - a * a);
        max_depth = max_depth * (1 - a * a);

        bool status = lambda1 > min_depth && lambda2 > min_depth;
        status = status && (lambda1 < max_depth) && (lambda2 < max_depth);
        return status;
    }

    torch::Tensor GetOrientationSignumTorch(const torch::Tensor& F,
                                            const torch::Tensor& epipole,
                                            const torch::Tensor& pt1,
                                            const torch::Tensor& pt2) {
        double signum1 = (F[0][0].item<double>() * pt2[0].item<double>() +
                          F[1][0].item<double>() * pt2[1].item<double>() +
                          F[2][0].item<double>());
        double signum2 = epipole[1].item<double>() -
                         epipole[2].item<double>() * pt1[1].item<double>();
        return torch::tensor(signum1 * signum2);
    }

    void EssentialFromMotionTorch(const Rigid3dTorch& pose, torch::Tensor* E) {
        // Create skew-symmetric matrix from translation
        auto t = pose.translation();
        auto t_cross = torch::zeros({3,3}, torch::kFloat64);
        t_cross[0][1] = -t[2].item<double>();
        t_cross[0][2] = t[1].item<double>();
        t_cross[1][0] = t[2].item<double>();
        t_cross[1][2] = -t[0].item<double>();
        t_cross[2][0] = -t[1].item<double>();
        t_cross[2][1] = t[0].item<double>();

        // E = [t]_x * R
        *E = torch::matmul(t_cross, pose.rotation_matrix());
    }

    void FundamentalFromMotionAndCamerasTorch(const Camera& camera1,
                                              const Camera& camera2,
                                              const Rigid3dTorch& pose,
                                              torch::Tensor* F) {
        // Get essential matrix
        torch::Tensor E;
        EssentialFromMotionTorch(pose, &E);

        // Convert camera matrices correctly, respecting column-major order
        const Eigen::Matrix3d& K1_eigen = camera1.GetK();
        const Eigen::Matrix3d& K2_eigen = camera2.GetK();

        // Create tensors with proper shape and order
        auto K1 = torch::empty({3, 3}, torch::kFloat64);
        auto K2 = torch::empty({3, 3}, torch::kFloat64);

        // Manual copy to ensure correct layout
        for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
                K1[i][j] = K1_eigen(i,j);
                K2[i][j] = K2_eigen(i,j);
            }
        }

        // Compute fundamental matrix: F = K2^-T * E * K1^-1
        auto K2_inv_t = K2.transpose(0,1).inverse();
        auto K1_inv = K1.inverse();

        *F = torch::matmul(torch::matmul(K2_inv_t, E), K1_inv);

        // Normalize to ensure consistent scale
        *F = *F / F->norm();
    }


    double SampsonError2DTorch(const torch::Tensor& E,
                               const torch::Tensor& x1,
                               const torch::Tensor& x2) {
        auto Ex1 = torch::matmul(E, torch_utils::to_homogeneous(x1).unsqueeze(-1)).squeeze();
        auto Etx2 = torch::matmul(E.transpose(0,1), torch_utils::to_homogeneous(x2).unsqueeze(-1)).squeeze();

        double C = torch_utils::to_double(torch::dot(Ex1, torch_utils::to_homogeneous(x2)));
        double Cx = torch_utils::to_double(torch::dot(Ex1.slice(0,0,2), Ex1.slice(0,0,2)));
        double Cy = torch_utils::to_double(torch::dot(Etx2.slice(0,0,2), Etx2.slice(0,0,2)));

        return C * C / (Cx + Cy);
    }

    double SampsonError3DTorch(const torch::Tensor& E,
                               const torch::Tensor& x1_3d,
                               const torch::Tensor& x2_3d) {
        // Match Eigen's normalization order
        auto Ex1 = torch::matmul(E, x1_3d.unsqueeze(-1)).squeeze() /
                   (torch_utils::to_double(x1_3d.index({2})) + EPS);
        auto Etx2 = torch::matmul(E.transpose(0,1), x2_3d.unsqueeze(-1)).squeeze() /
                    (torch_utils::to_double(x2_3d.index({2})) + EPS);

        double C = torch_utils::to_double(torch::dot(Ex1, x2_3d));
        double Cx = torch_utils::to_double(torch::dot(Ex1.slice(0,0,2), Ex1.slice(0,0,2)));
        double Cy = torch_utils::to_double(torch::dot(Etx2.slice(0,0,2), Etx2.slice(0,0,2)));

        return C * C / (Cx + Cy);
    }

    double HomographyErrorTorch(const torch::Tensor& H,
                                const torch::Tensor& x1,
                                const torch::Tensor& x2) {
        auto Hx1 = torch::matmul(H, torch_utils::to_homogeneous(x1).unsqueeze(-1)).squeeze();
        auto z = torch_utils::to_double(Hx1.index({2}));
        auto Hx1_norm = Hx1.slice(0,0,2) / (z + EPS);

        return torch_utils::to_double((Hx1_norm - x2).pow(2).sum());
    }

} // namespace glomap