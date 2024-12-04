// rigid3d_torch_test.cc
#include "glomap/math/rigid3d_torch_utils.h"
#include <gtest/gtest.h>

namespace glomap {
    namespace {

        // Test helper class - moved here since it needs gtest
        class RigidTester {
        public:
            static void ExpectTransformsEqual(const Rigid3d& eigen_transform,
                                              const Rigid3dTorch& torch_transform,
                                              const std::string& message = "") {
                EXPECT_TRUE(RigidComparator::IsApproxEqual(eigen_transform, torch_transform))
                    << "Transforms are not equal: " << message;
                EXPECT_TRUE(RigidComparator::HasApproxEqualEffect(eigen_transform, torch_transform))
                    << "Transforms have different effects: " << message;
            }

            // Generate random valid transform for testing
            static std::pair<Rigid3d, Rigid3dTorch> GenerateRandomTransform() {
                // Create random rotation
                Eigen::Vector3d axis = Eigen::Vector3d::Random().normalized();
                double angle = (rand() % 360) * M_PI / 180.0;
                Eigen::Quaterniond rotation(Eigen::AngleAxisd(angle, axis));

                // Create random translation
                Eigen::Vector3d translation = Eigen::Vector3d::Random();

                // Create both versions
                Rigid3d eigen_transform(rotation, translation);
                Rigid3dTorch torch_transform = RigidConverter::EigenToTorch(eigen_transform);

                return {eigen_transform, torch_transform};
            }
        };

        TEST(Rigid3dTorch, IdentityTransform) {
            Rigid3d eigen_identity;
            Rigid3dTorch torch_identity;

            RigidTester::ExpectTransformsEqual(eigen_identity, torch_identity,
                                               "Identity transforms should be equal");
        }

        TEST(Rigid3dTorch, RandomTransforms) {
            for (int i = 0; i < 100; ++i) {
                auto [eigen_transform, torch_transform] = RigidTester::GenerateRandomTransform();
                RigidTester::ExpectTransformsEqual(eigen_transform, torch_transform,
                                                   "Random transform " + std::to_string(i));
            }
        }

        TEST(Rigid3dTorch, Composition) {
            // Generate two random transforms
            auto [eigen_t1, torch_t1] = RigidTester::GenerateRandomTransform();
            auto [eigen_t2, torch_t2] = RigidTester::GenerateRandomTransform();

            // Compose them
            Rigid3d eigen_result = eigen_t1 * eigen_t2;
            Rigid3dTorch torch_result = torch_t1 * torch_t2;

            RigidTester::ExpectTransformsEqual(eigen_result, torch_result,
                                               "Composed transforms should be equal");
        }

        TEST(Rigid3dTorch, Inverse) {
            auto [eigen_transform, torch_transform] = RigidTester::GenerateRandomTransform();

            Rigid3d eigen_inverse = Inverse(eigen_transform);
            Rigid3dTorch torch_inverse = torch_transform.inverse();

            RigidTester::ExpectTransformsEqual(eigen_inverse, torch_inverse,
                                               "Inverse transforms should be equal");

            // Test that composition with inverse gives identity
            Rigid3d eigen_identity;
            Rigid3dTorch torch_identity;

            RigidTester::ExpectTransformsEqual(
                eigen_transform * eigen_inverse,
                torch_transform * torch_inverse,
                "Transform * inverse should equal identity");
        }

        TEST(Rigid3dTorch, AngleCalculation) {
            for (int i = 0; i < 100; ++i) {
                auto [eigen_t1, torch_t1] = RigidTester::GenerateRandomTransform();
                auto [eigen_t2, torch_t2] = RigidTester::GenerateRandomTransform();

                double eigen_angle = CalcAngle(eigen_t1, eigen_t2);
                double torch_angle = calc_angle(torch_t1, torch_t2);

                EXPECT_NEAR(eigen_angle, torch_angle, kConversionTolerance)
                    << "Angles should be equal for random transforms " << i;
            }
        }

    } // namespace
} // namespace glomap