//============================================================================
//
// This file is part of the ShapePFCN project.
//
// Copyright (c) 2016-2017 - Evangelos Kalogerakis, Melinos Averkiou, Siddhartha Chaudhuri, Subhransu Maji
//
// ShapePFCN is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// ShapePFCN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with ShapePFCN.  If not, see <http://www.gnu.org/licenses/>.
//
//============================================================================

#ifndef __MVFCN_RenderViews_hpp__
#define __MVFCN_RenderViews_hpp__

#include "Common.hpp"
#include "MeshProcessor.hpp"
#include "ShapeRenderer.hpp"

using namespace Thea;
using namespace Algorithms;
using namespace Graphics;

class RenderViews
{

public:
  RenderViews(const std::shared_ptr<MeshProcessor>& _mesh_processor_ptr, const string& _dataset_path, const std::map<string, int>& _label_map, int _camera_starting_index = 0);

	void render();

private:
  int camera_starting_index;
	std::shared_ptr<MeshProcessor> mesh_processor_ptr;
	std::map<string, int> label_map;

	string dataset_path;
	int num_samples;
	Vector3 up;
	Vector2 image_size;
	bool flat_shading;

	TheaArray<Vector3> samples_positions;
	TheaArray<Vector3> samples_normals;
	TheaArray<long> samples_face_indices;

	void sample_points();

	void dodecahedron_points();

	void save_camera_distances(const TheaArray<Real>& camera_distances);

	void save_sample_points();

	bool renderFaceIndexImage(ShapeRenderer& renderer, std::string& cmd);

	bool renderAndSaveFaceIndexImage(ShapeRenderer& renderer, std::string& cmd, size_t point_index, size_t cam_dist_index, size_t upv_index, size_t render_index);

	bool renderAndSaveRemainingImages(ShapeRenderer& renderer, std::string& cmd, size_t point_index, size_t cam_dist_index, size_t upv_index, size_t render_index);

	void find_visible_points(TheaSet<uint32>& visible_point_indices, const Image& face_ids_image,const Vector3& view_vector);

	TheaArray<Thea::Real> getCameraDistances(float fov);
};

class Viewpoint
{

public:

	size_t point_index;
	size_t cam_dist_index;

	TheaSet<uint32> visible_point_indices;

	bool operator< (const Viewpoint &other) const 
	{
		return visible_point_indices.size() < other.visible_point_indices.size();
	}

	bool operator> (const Viewpoint &other) const
	{
		return visible_point_indices.size() > other.visible_point_indices.size();
	}
};
#endif