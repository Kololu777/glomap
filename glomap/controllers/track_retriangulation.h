
#pragma once

#include "glomap/scene/types_sfm.h"
#include <glomap/colmap_migration/database.h>

namespace glomap {

    struct TriangulatorOptions {
        double tri_complete_max_reproj_error = 15.0;
        double tri_merge_max_reproj_error = 15.0;
        double tri_min_angle = 1.0;

        int min_num_matches = 15;
    };

    bool RetriangulateTracks(const TriangulatorOptions& options,
                             const Database& database,
                             std::unordered_map<camera_t, Camera>& cameras,
                             std::unordered_map<image_t, migration::Image>& images,
                             std::unordered_map<track_t, migration::Track>& tracks);

} // namespace glomap
