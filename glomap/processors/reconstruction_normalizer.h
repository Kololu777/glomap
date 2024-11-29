#pragma once

#include "colmap/geometry/pose.h"
#include "glomap/scene/types_sfm.h"

namespace glomap {

    colmap::Sim3d NormalizeReconstruction(
        std::unordered_map<camera_t, Camera>& cameras,
        std::unordered_map<image_t, Image>& images,
        std::unordered_map<track_t, Track>& tracks,
        bool fixed_scale = false,
        double extent = 10.,
        double p0 = 0.1,
        double p1 = 0.9);
} // namespace glomap
