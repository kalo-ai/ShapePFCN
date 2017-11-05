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

#include "RenderViews.hpp"

RenderViews::RenderViews(const std::shared_ptr<MeshProcessor>& _mesh_processor_ptr, const string& _dataset_path, const std::map<string, int>& _label_map, int _camera_starting_index)
{
	assert(_mesh_processor_ptr != NULL);
	assert(_mesh_processor_ptr->getMeshPtr() != NULL);
	assert(!_mesh_processor_ptr->getMeshPtr()->isEmpty());
	assert(!_mesh_processor_ptr->getMeshPath().empty());

  camera_starting_index = _camera_starting_index;
	mesh_processor_ptr = _mesh_processor_ptr;
	dataset_path = _dataset_path;
	label_map = _label_map;
	num_samples = Settings::num_sample_points;
  up = Settings::up_vector;
	image_size = Vector2(Settings::render_size,Settings::render_size);
	flat_shading = Settings::flat_shading;
}



void RenderViews::sample_points()
{
  if (load_sample_points())
    return;

	MeshSampler<Mesh> sampler(*(mesh_processor_ptr->getMeshPtr()));
	TheaArray< MeshSampler<Mesh>::Triangle const * > tris;
	samples_positions.clear();
	samples_normals.clear();
	samples_face_indices.clear();
	sampler.sampleEvenlyBySeparation(num_samples, samples_positions, &samples_normals, &tris,
		MeshSampler<Mesh>::CountMode::EXACT, -1, true);

	alwaysAssertM(tris.size() >= samples_positions.size(), "Triangle ID's not initialized");

	samples_face_indices.resize(samples_positions.size());
	for (array_size_t i = 0; i < samples_positions.size(); ++i)
	{
		Mesh::Face const * face = tris[i]->getVertices().getMeshFace();
		samples_face_indices[i] = face->attr().index;
	}

	//MeshFeatures::Local::ShapeDiameter<Mesh> sdf(*(mesh_processor_ptr->getMeshPtr()));

	// Sanitize outputs
	for (array_size_t i = 0; i < samples_positions.size(); ++i)
	{
		if (std::fabs(samples_positions[i].x()) < 1e-35) samples_positions[i].x() = 0;
		if (std::fabs(samples_positions[i].y()) < 1e-35) samples_positions[i].y() = 0;
		if (std::fabs(samples_positions[i].z()) < 1e-35) samples_positions[i].z() = 0;

		if (std::fabs(samples_normals[i].x()) < 1e-35) samples_normals[i].x() = 0;
		if (std::fabs(samples_normals[i].y()) < 1e-35) samples_normals[i].y() = 0;
		if (std::fabs(samples_normals[i].z()) < 1e-35) samples_normals[i].z() = 0;

		samples_normals[i].fastUnitize();

		// Check whether we need to flip the normal
		//double sdf_value_normal = sdf.compute(samples_positions[i], samples_normals[i], true);
		//double sdf_value_flipped = sdf.compute(samples_positions[i], -samples_normals[i], true);

		//if (sdf_value_normal < 0)
		//{
		//	// Flip normal since sdf for the normal direction returned negative value, meaning no intersections
		//	samples_normals[i] = -samples_normals[i];
		//}
		//else if (sdf_value_flipped > 0)
		//{
		//	// sdf for the normal direction returned positive value and sdf for the flipped normal direction also returned positive value
		//	// so take the smallest sdf value to be the right normal
		//	if (sdf_value_flipped < sdf_value_normal)
		//		samples_normals[i] = -samples_normals[i];
		//}

	}

}

void RenderViews::save_camera_distances(const TheaArray<Real>& camera_distances)
{
	// Save cam distances in a file in the same folder as the mesh
  std::string cam_dists_file;
  if (Settings::baseline_rendering)
    cam_dists_file = FilePath::concat(FilePath::concat(dataset_path, LEARNING_METADATA_FOLDER), FilePath::baseName(mesh_processor_ptr->getMeshPath()) + "_cam_dists_baseline.txt");
  else
    cam_dists_file = FilePath::concat(FilePath::concat(dataset_path, LEARNING_METADATA_FOLDER), FilePath::baseName(mesh_processor_ptr->getMeshPath()) + "_cam_dists.txt");

	ofstream cam_out(cam_dists_file.c_str());
	if (cam_out)
	{
		for (TheaArray<Real>::const_iterator it = camera_distances.begin(); it != camera_distances.end(); ++it)
		{
			cam_out << *it << std::endl;
		}
	}
	cam_out.close();
}

void RenderViews::save_sample_points()
{
	// Save the sampled points and their normals in a file in the same folder as the mesh
  std::string points_file;
  if (Settings::baseline_rendering)
	  points_file = FilePath::concat(FilePath::concat(dataset_path, MESH_METADATA_FOLDER), FilePath::baseName(mesh_processor_ptr->getMeshPath()) + "_points_baseline.txt");
  else
    points_file = FilePath::concat(FilePath::concat(dataset_path, MESH_METADATA_FOLDER), FilePath::baseName(mesh_processor_ptr->getMeshPath()) + "_points.txt");

  Model* model = mesh_processor_ptr->getModel();
	ofstream points_out(points_file.c_str());
	if (points_out)
	{
		int ind = 0;
    TheaArray<Vector3>::iterator itn = model->sample_normals.begin();
    for (TheaArray<Vector3>::iterator it = model->sample_points.begin(); it != model->sample_points.end(); ++it, ++itn)
		{
      points_out << (*it)[0] << " " << (*it)[1] << " " << (*it)[2] << " " << (*itn)[0] << " " << (*itn)[1] << " " << (*itn)[2] << std::endl;
			ind++;
		}
	}
	points_out.close();
}

bool RenderViews::load_sample_points()
{
  // Save the sampled points and their normals in a file in the same folder as the mesh
  std::string points_file;
  if (Settings::baseline_rendering)
    points_file = FilePath::concat(FilePath::concat(dataset_path, MESH_METADATA_FOLDER), FilePath::baseName(mesh_processor_ptr->getMeshPath()) + "_points_baseline.txt");
  else
    points_file = FilePath::concat(FilePath::concat(dataset_path, MESH_METADATA_FOLDER), FilePath::baseName(mesh_processor_ptr->getMeshPath()) + "_points.txt");

  ifstream points_in(points_file.c_str());
  if (!points_in.good() && Settings::baseline_rendering)
  {
    // make a second attempt to find pts files
    points_file = FilePath::concat(dataset_path, FilePath::baseName(mesh_processor_ptr->getMeshPath()) + ".pts");
    points_in.open(points_file.c_str());
    if (!points_in.good())
      return false;

    float x, y, z;
    while (!points_in.eof())
    {
      points_in >> x >> y >> z;

      Thea::Vector3 pos = Thea::Vector3(x, y, z);
      samples_positions.push_back(pos);
      samples_normals.push_back(Thea::Vector3(0.0, 0.0, 0.0));
    }

    THEA_CONSOLE << "Found sample points file for input mesh (pts file) - read surface sample points from it - no normals, however!";

    points_in.close();
    return true;
  }
  if (!points_in.good())
    return false;

  float x, y, z, nx, ny, nz;
  while (!points_in.eof())
  {
    points_in >> x >> y >> z;
    if (points_in.eof())
      break;

    points_in >> nx >> ny >> nz;

    Thea::Vector3 pos = Thea::Vector3(x, y, z);
    samples_positions.push_back( pos );
    Thea::Vector3 normal = Thea::Vector3(nx, ny, nz);
    normal.fastUnitize();
    samples_normals.push_back(normal);
  }

  THEA_CONSOLE << "Found sample points and normals file for input mesh - read surface sample points from it!";

  points_in.close();
  return true;
}

void RenderViews::dodecahedron_points()
{
	samples_positions.clear();
	samples_normals.clear();
	samples_face_indices.clear();
	float phi = (1 + sqrt(5)) / 2;

	//Flipped y with z compared to renderviews.m
	samples_positions.push_back(Thea::Vector3(1, 1, 1));
	samples_positions.push_back(Thea::Vector3(1, -1, 1));
	samples_positions.push_back(Thea::Vector3(1, 1, -1));
	samples_positions.push_back(Thea::Vector3(1, -1, -1));
	samples_positions.push_back(Thea::Vector3(-1, 1, 1));
	samples_positions.push_back(Thea::Vector3(-1, -1, 1));
	samples_positions.push_back(Thea::Vector3(-1, 1, -1));
	samples_positions.push_back(Thea::Vector3(-1, -1, -1));
	samples_positions.push_back(Thea::Vector3(0, phi, 1 / phi));
	samples_positions.push_back(Thea::Vector3(0, -phi, 1 / phi));
	samples_positions.push_back(Thea::Vector3(0, phi, -1 / phi));
	samples_positions.push_back(Thea::Vector3(0, -phi, -1 / phi));
	samples_positions.push_back(Thea::Vector3(phi, 1 / phi, 0));
	samples_positions.push_back(Thea::Vector3(phi, -1 / phi, 0));
	samples_positions.push_back(Thea::Vector3(-phi, 1 / phi, 0));
	samples_positions.push_back(Thea::Vector3(-phi, -1 / phi, 0));
	samples_positions.push_back(Thea::Vector3(1 / phi, 0, phi));
	samples_positions.push_back(Thea::Vector3(-1 / phi, 0, phi));
	samples_positions.push_back(Thea::Vector3(1 / phi, 0, -phi));
	samples_positions.push_back(Thea::Vector3(-1 / phi, 0, -phi));

}

TheaArray<Thea::Real> RenderViews::getCameraDistances(float fov)
{
	THEA_CONSOLE << "Computing camera distances for mesh " << mesh_processor_ptr->getMeshPtr()->getName();

	TheaArray<Thea::Real> camera_distances;

	if (Settings::baseline_rendering)
	{
		float mesh_radius = mesh_processor_ptr->getMeshRadius();
		mesh_radius = mesh_radius*1.01;
		camera_distances.push_back(mesh_radius / tan(fov / 2.0f));
	}
	else
	{
		// Find the maximum euclidean distance between any two sample points 
		float max_sample_distance = 0.0f;
		for (TheaArray<Vector3>::iterator it = samples_positions.begin(); it != samples_positions.end(); ++it)
		{
			for (TheaArray<Vector3>::iterator it2 = samples_positions.begin(); it2 != samples_positions.end(); ++it2)
			{
				float sqdist_ij = (*it - *it2).squaredLength();
				if (sqdist_ij > max_sample_distance)
				{
					max_sample_distance = sqdist_ij;
				}
			}
		}
		max_sample_distance = sqrt(max_sample_distance);
		max_sample_distance = max_sample_distance / 2.0;
		//max_sample_distance = max_sample_distance*1.01;

		float r = max_sample_distance / tan(fov / 2.0f);

    float cam_step = 1.5f / ( (float)Settings::num_cam_distances + 1.0f ); // Maybe we want to go further than r? add a parameter for min-max? V: let's have +50% slack

		for (int i = 1; i <= Settings::num_cam_distances; ++i)
		{
			camera_distances.push_back(i*cam_step*r);
		}
	}
	return camera_distances;
}

void RenderViews::render()
{
	// Sample points
	if (Settings::baseline_rendering)
	{
    Model* model = mesh_processor_ptr->getModel();
    sample_points();
    model->sample_points = samples_positions;
    model->sample_normals = samples_normals;
		dodecahedron_points();
	}
	else
	{
		sample_points();

		if (samples_positions.empty() || samples_normals.empty() || samples_face_indices.empty())
		{
			THEA_ERROR << "Cannot sample points on mesh " << mesh_processor_ptr->getMeshPath();
			return;
		}
	}

	// Get list of camera distances
	TheaArray<Real> camera_distances = getCameraDistances(Settings::fov);

	if (camera_distances.empty())
	{
		THEA_ERROR << "Cannot get list of camera distances on mesh " << mesh_processor_ptr->getMeshPath();
		return;
	}

	// Save camera distances and sample points
	save_camera_distances(camera_distances);
	save_sample_points();

	// Render
	THEA_CONSOLE << "Rendering...";
	try
	{
		Model* model = mesh_processor_ptr->getModel();

		if (Settings::baseline_rendering)
		{
			////careful, sample_positions for the baseline_rendering case are not on the mesh surface so the point_id image will make no sense
      // this is not needed
			//model->sample_points.clear();
			//model->sample_normals.clear();
		}
		else
		{
			model->sample_points = samples_positions;
			model->sample_normals = samples_normals;
		}
		TheaArray<Viewpoint> out_viewpoints; // for non-baseline rendering (algorithm_v4)
		int render_index = 0;
		for (size_t i = 0; i < camera_distances.size(); ++i)
		{
			Real dist_i = camera_distances[i];

			TheaArray<Viewpoint> viewpoints; // for non-baseline rendering (algorithm_v4)
//#pragma omp parallel for
			for (long p = 0; p < samples_positions.size(); ++p)
			{
				ShapeRenderer renderer(mesh_processor_ptr->getMeshPath(), *model);

				if (Settings::baseline_rendering)
				{
					Vector3 sample_pos = samples_positions[p];
					sample_pos.fastUnitize();
					Vector3 mesh_center = mesh_processor_ptr->getMeshCenter();
					Vector3 current_cam_pos = mesh_center + sample_pos*dist_i;
					Vector3 look_vector = mesh_center - current_cam_pos;
					look_vector.fastUnitize();

          Vector3 right;
          bool do_inplane_rotation = false;
          if (abs(look_vector.dot(up)) < 0.97) //0.97 is kind of arbitrary, we want 1.0 but add some epsilon to it
          {
            right = look_vector.cross(up); // use up-vector to determine right vector
          }
          else // use one of the canonical axis
          {
            do_inplane_rotation = true;
            if (abs(look_vector.dot(Vector3(1, 0, 0))) < 0.97)
              right = look_vector.cross(Vector3(1, 0, 0));
            else if (abs(look_vector.dot(Vector3(0, 1, 0))) < 0.97)
              right = look_vector.cross(Vector3(0, 1, 0));
            else
              right = look_vector.cross(Vector3(0, 0, 1));
          }

					right.fastUnitize();
					Vector3 up1 = right.cross(look_vector);

					TheaArray<Vector3> up_vecs;
					up_vecs.push_back(up1);
          if (!Settings::use_upright_coord || do_inplane_rotation)
          {
            up_vecs.push_back(Matrix3::rotationAxisAngle(look_vector, 90.0f *M_PI / 180.0f) * up1);
            up_vecs.push_back(Matrix3::rotationAxisAngle(look_vector, 180.0f *M_PI / 180.0f) * up1);
            up_vecs.push_back(Matrix3::rotationAxisAngle(look_vector, 270.0f *M_PI / 180.0f) * up1);
          }
					// Render for each of the four up vectors
					for (size_t u = 0; u < up_vecs.size(); ++u)
					{
						ostringstream ss;
						ss.str("");
						ss.clear();
						ss << look_vector.x() << "," << look_vector.y() << "," << look_vector.z() \
							<< "," << dist_i << "," << current_cam_pos.x() << "," << current_cam_pos.y() << "," << current_cam_pos.z() \
							<< "," << up_vecs[u].x() << "," << up_vecs[u].y() << "," << up_vecs[u].z() << " " << image_size.x() << " " << image_size.y(); 
						string cmd = ss.str();

						if (!renderAndSaveFaceIndexImage(renderer, cmd, p, i, u, render_index))
						{
							continue;
						}
						// Render the remaining images
						renderAndSaveRemainingImages(renderer, cmd, p, i, u, render_index);
					}
					render_index++;
				}
				else // non-baseline rendering (algorithm_v4)
				{
					Vector3 sample_pos = samples_positions[p];
					Vector3 sample_normal = samples_normals[p];
					Vector3 current_cam_pos = sample_pos + sample_normal*dist_i;
					Vector3 look_vector = -sample_normal;
					look_vector.fastUnitize();

          Vector3 right;
          if (abs(look_vector.dot(up)) < 0.97) //0.97 is kind of arbitrary, we want 1.0 but add some epsilon to it
          {
            right = look_vector.cross(up); // use up-vector to determine right vector
          }
          else // use one of the canonical axis
          {
            if (abs(look_vector.dot(Vector3(1, 0, 0))) < 0.97)
              right = look_vector.cross(Vector3(1, 0, 0));
            else if (abs(look_vector.dot(Vector3(0, 1, 0))) < 0.97)
              right = look_vector.cross(Vector3(0, 1, 0));
            else
              right = look_vector.cross(Vector3(0, 0, 1));
          }

					right.fastUnitize();
					Vector3 up1 = right.cross(look_vector);

					ostringstream ss;
					ss << look_vector.x() << "," << look_vector.y() << "," << look_vector.z() \
						<< "," << dist_i << "," << current_cam_pos.x() << "," << current_cam_pos.y() << "," << current_cam_pos.z() \
						<< "," << up1.x() << "," << up1.y() << "," << up1.z() << " " << image_size.x() << " " << image_size.y();
					string cmd = ss.str();

					if (!renderFaceIndexImage(renderer, cmd))
					{
						continue;
					}

					Viewpoint vp;
          vp.cam_dist_index = i;
					vp.point_index = p;

					Image& face_point_ids_image = renderer.get_color_image();

					find_visible_points(vp.visible_point_indices, face_point_ids_image,sample_normal);
					viewpoints.push_back(vp);
				}
			}

			if (!Settings::baseline_rendering) // non-baseline rendering (algorithm_v4)
			{
				TheaSet<uint32> covered_point_indices;
				int num_selected_views = 0;
				// repeat as long as we haven't covered all the sample points
				while (covered_point_indices.size() != samples_positions.size())
				{	
					// Go over all viewpoints and for each one update its set of visible points according to the points covered so far
					for (TheaArray<Viewpoint>::iterator viewpoint_it = viewpoints.begin(); viewpoint_it != viewpoints.end(); ++viewpoint_it)
					{
						TheaSet<uint32>::iterator visible_point_it = viewpoint_it->visible_point_indices.begin();
						// Go over all visible points for this viewpoint and remove those who are already covered
						for (; visible_point_it != viewpoint_it->visible_point_indices.end(); ) 
						{
							if (covered_point_indices.find(*visible_point_it) != covered_point_indices.end())
							{
								visible_point_it = viewpoint_it->visible_point_indices.erase(visible_point_it);
							}
							else 
							{
								++visible_point_it;
							}
						}
					}

					// Sort the viewpoints in descending order according to the size of their visible point sets (i.e. uncovered area that they cover)
					std::sort(viewpoints.begin(), viewpoints.end(),std::greater<Viewpoint>());

					// Copy all the visible points of the top (after sorting) viewpoint to the set of covered points
					covered_point_indices.insert(viewpoints.begin()->visible_point_indices.begin(), viewpoints.begin()->visible_point_indices.end());
					
					out_viewpoints.push_back(*viewpoints.begin());
					num_selected_views++;
					// Stop adding viewpoints from this camera distance if we reached the maximum
					if (num_selected_views >= Settings::max_images_per_distance)
					{
						break;
					}
					// Remove the top viewpoint from the set of viewpoints
					viewpoints.erase(viewpoints.begin());
				}
			}
		}

		if (!Settings::baseline_rendering) // non-baseline rendering (algorithm_v4)
		{
			//TheaArray<Viewpoint>::iterator out_viewpoints_it = out_viewpoints.begin(); //omp doesn't like iterators
			//TheaArray<Viewpoint>::iterator out_viewpoints_end = out_viewpoints.end();  //omp doesn't like iterators
//#pragma omp parallel for
			// Go over all selected viewpoints and do the actual rendering of all images (for 4 rotations of the up vector)
			//for (; out_viewpoints_it != out_viewpoints_end; ++out_viewpoints_it) //omp doesn't like iterators
			for (long v = 0; v < out_viewpoints.size(); ++v)
			{
				ShapeRenderer renderer(mesh_processor_ptr->getMeshPath(), *model);

				Viewpoint& current_viewpoint = out_viewpoints[v]; //*out_viewpoints_it; //omp doesn't like iterators
				size_t p = current_viewpoint.point_index;
				size_t i = current_viewpoint.cam_dist_index;
				Real dist_i = camera_distances[i];

				Vector3 sample_pos = samples_positions[current_viewpoint.point_index];
				Vector3 sample_normal = samples_normals[current_viewpoint.point_index];
				Vector3 current_cam_pos = sample_pos + sample_normal*dist_i;
				Vector3 look_vector = -sample_normal;
				look_vector.fastUnitize();

        Vector3 right;
        bool do_inplane_rotation = false;
        if (abs(look_vector.dot(up)) < 0.97) //0.97 is kind of arbitrary, we want 1.0 but add some epsilon to it
        {
          right = look_vector.cross(up); // use up-vector to determine right vector
        }
        else // use one of the canonical axis
        {
          do_inplane_rotation = true;
          if (abs(look_vector.dot(Vector3(1, 0, 0))) < 0.97)
            right = look_vector.cross(Vector3(1, 0, 0));
          else if (abs(look_vector.dot(Vector3(0, 1, 0))) < 0.97)
            right = look_vector.cross(Vector3(0, 1, 0));
          else
            right = look_vector.cross(Vector3(0, 0, 1));
        }

				right.fastUnitize();
				Vector3 up1 = right.cross(look_vector);

				TheaArray<Vector3> up_vecs;
				up_vecs.push_back(up1);
        if (!Settings::use_upright_coord || do_inplane_rotation)
        {
          up_vecs.push_back(Matrix3::rotationAxisAngle(look_vector, 90.0f *M_PI / 180.0f) * up1);
          up_vecs.push_back(Matrix3::rotationAxisAngle(look_vector, 180.0f *M_PI / 180.0f) * up1);
          up_vecs.push_back(Matrix3::rotationAxisAngle(look_vector, 270.0f *M_PI / 180.0f) * up1);
        }
				// Render for each of the four up vectors
				for (size_t u = 0; u < up_vecs.size(); ++u)
				{
					ostringstream ss;
					ss.str("");
					ss.clear();
					ss << look_vector.x() << "," << look_vector.y() << "," << look_vector.z() \
						<< "," << dist_i << "," << current_cam_pos.x() << "," << current_cam_pos.y() << "," << current_cam_pos.z() \
						<< "," << up_vecs[u].x() << "," << up_vecs[u].y() << "," << up_vecs[u].z() << " " << image_size.x() << " " << image_size.y();
					string cmd = ss.str();

					if (!renderAndSaveFaceIndexImage(renderer, cmd, p, i, u, v))
					{
						continue;
					}
					// Render the remaining images
					renderAndSaveRemainingImages(renderer, cmd, p, i, u, v);
				}
			}
		}
	}
	THEA_STANDARD_CATCH_BLOCKS(return;, ERROR, "%s", "Could not create shape renderer");

  THEA_CONSOLE << "Rendering main pass complete."; 
}

// Render only, don't save the image (since we use this in the algorithm_v4 method)
bool RenderViews::renderFaceIndexImage(ShapeRenderer& renderer, std::string& cmd)
{
	// Render FIRST both faces and sample points using their ids (indices)
	string cmd_line = "render -c id -v ";
	cmd_line += cmd;

	if (flat_shading)
		cmd_line += " -f";

	int status = renderer.exec(cmd_line);

	if (status == -1)
	{
		THEA_ERROR << "Could not render the face and point indices image for " << mesh_processor_ptr->getMeshPath();
		return false;
	}

	return true;
}

// Render AND save the image (for the baseline)
bool RenderViews::renderAndSaveFaceIndexImage(ShapeRenderer& renderer, std::string& cmd, size_t point_index, size_t cam_dist_index, size_t upv_index, size_t render_index)
{
	if (!renderFaceIndexImage(renderer, cmd))
		return false;

	// Save the face and point ids image
	Image& face_point_ids_image = renderer.get_color_image();

  std::string fpid_image_name = FilePath::concat(FilePath::concat(dataset_path, TRIANGLEID_IMAGES_FOLDER), FilePath::baseName(mesh_processor_ptr->getMeshPath()) + format("_fpid_%06ld_%06ld_%06ld_%06ld", render_index, point_index, cam_dist_index + camera_starting_index, upv_index));
  if (mesh_processor_ptr->isUsedForValidation())
    fpid_image_name += "@.png";
  else
    fpid_image_name += ".png";

	face_point_ids_image.save(fpid_image_name);



	return true;
}

// Render AND save all the remaining images (point and cam_dist index are required for saving the images)
bool RenderViews::renderAndSaveRemainingImages(ShapeRenderer& renderer, std::string& cmd, size_t point_index, size_t cam_dist_index, size_t upv_index, size_t render_index)
{
	std::string cmd2 = cmd;
	if (flat_shading)
		cmd2 += " -f";

	// Render SECOND only faces using their ids (indices)
	string cmd_line2 = "render -c fid -v ";
	cmd_line2 += cmd2;
	int status2 = renderer.exec(cmd_line2);

	if (status2 == -1)
	{
		THEA_ERROR << "Could not render the face indices image for " << mesh_processor_ptr->getMeshPath();
		return false;
	}

	// Save the face ids image
  std::string fid_image_name = FilePath::concat(FilePath::concat(dataset_path, TRIANGLEID_IMAGES_FOLDER), FilePath::baseName(mesh_processor_ptr->getMeshPath()) + format("_fid_%06ld_%06ld_%06ld_%06ld", render_index, point_index, cam_dist_index + camera_starting_index, upv_index));
  if (mesh_processor_ptr->isUsedForValidation())
    fid_image_name += "@.png";
  else
    fid_image_name += ".png";

	Image& face_ids_image = renderer.get_color_image();

	// Save the face ids image
	face_ids_image.save(fid_image_name);

	///////////////////////////////////////

	// Render THIRD only faces using their face labels
	string cmd_line3 = "render -c lbl -v ";
	cmd_line3 += cmd2;
	int status3 = renderer.exec(cmd_line3);

	if (status3 == -1)
	{
		THEA_ERROR << "Could not render the labels image for " << mesh_processor_ptr->getMeshPath();
		return false;
	}

	// Save the labels image
  std::string lbl_image_name = FilePath::concat(FilePath::concat(dataset_path, SEGMENTATION_IMAGES_FOLDER), FilePath::baseName(mesh_processor_ptr->getMeshPath()) + format("_lbl_%06ld_%06ld_%06ld_%06ld", render_index, point_index, cam_dist_index + camera_starting_index, upv_index));
  if (mesh_processor_ptr->isUsedForValidation())
    lbl_image_name += "@.png";
  else
    lbl_image_name += ".png";


	Image& lbl_image = renderer.get_color_image();
	lbl_image.convert(Thea::Image::Type::LUMINANCE_8U); // V addition, convert to grayscale [assuming Red=Green=Blue here, as it is set by the modified indexToColor in ShapeRenderer.cpp]

														// Save the labels image
	lbl_image.save(lbl_image_name);

	///////////////////////////////////////

	// Render THIRD ALTERNATIVE: only faces using their face labels - USE COLOR PALLETE HERE
	string cmd_line3b = "render -c lblp -v ";
	cmd_line3b += cmd2;
	int status3b = renderer.exec(cmd_line3b);

	if (status3b == -1)
	{
		THEA_ERROR << "Could not render the labels image with pallette for " << mesh_processor_ptr->getMeshPath();
		return false;
	}

	// Save the labels image
  std::string lblp_image_name = FilePath::concat(FilePath::concat(dataset_path, SEGMENTATION_COLOR_IMAGE_FOLDER), FilePath::baseName(mesh_processor_ptr->getMeshPath()) + format("_lblp_%06ld_%06ld_%06ld_%06ld", render_index, point_index, cam_dist_index + camera_starting_index, upv_index));
  if (mesh_processor_ptr->isUsedForValidation())
    lblp_image_name += "@.png";
  else
    lblp_image_name += ".png";

	Image& lblp_image = renderer.get_color_image();

	// Save the labels image
	lblp_image.save(lblp_image_name);

	///////////////////////////////////////

	//Render FOURTH the proper color and depth images (render with gray)
  if (!Settings::use_consistent_coord)
  {
	  string cmd_line4 = "render -c gray -v ";
	  cmd_line4 += cmd2;
	  int status4 = renderer.exec(cmd_line4);
  
  	if (status4 == -1)
  	{
		  THEA_ERROR << "Could not render the color and depth image for " << mesh_processor_ptr->getMeshPath();
		  return false;
  	}

    // Save the color and depth images
    std::string color_image_name = FilePath::concat(FilePath::concat(dataset_path, PRENDERED_IMAGES_FOLDER), FilePath::baseName(mesh_processor_ptr->getMeshPath()) + format("_int_%06ld_%06ld_%06ld_%06ld", render_index, point_index, cam_dist_index + camera_starting_index, upv_index));
    if (mesh_processor_ptr->isUsedForValidation())
      color_image_name += "@.png";
    else
      color_image_name += ".png";
    Image& color_image = renderer.get_color_image();
    color_image.save(color_image_name);

    std::string depth_image_name = FilePath::concat(FilePath::concat(dataset_path, DEPTH_IMAGES_FOLDER), FilePath::baseName(mesh_processor_ptr->getMeshPath()) + format("_dep_%06ld_%06ld_%06ld_%06ld", render_index, point_index, cam_dist_index + camera_starting_index, upv_index));
    if (mesh_processor_ptr->isUsedForValidation())
      depth_image_name += "@.png";
    else
      depth_image_name += ".png";

    Image& depth_image = renderer.get_depth_image();
    depth_image.save(depth_image_name);
  }
  else
  {
    // CONSISTENT ORIENTATION change starts
    //Render THIRD x-image
    Thea::Vector3 tmp = Settings::up_vector;
    Settings::up_vector = Thea::Vector3(1, 0, 0);

    string cmd_line4 = "render -c aux -v ";
    cmd_line4 += cmd2;
    int status4 = renderer.exec(cmd_line4);

    if (status4 == -1)
    {
      THEA_ERROR << "Could not render the x-image for " << mesh_processor_ptr->getMeshPath();
      return false;
    }

    std::string aux_image_name = FilePath::concat(FilePath::concat(dataset_path, PRENDERED_IMAGES_FOLDER), FilePath::baseName(mesh_processor_ptr->getMeshPath()) + format("_int_%06ld_%06ld_%06ld_%06ld", render_index, point_index, cam_dist_index + camera_starting_index, upv_index));
    if (mesh_processor_ptr->isUsedForValidation())
      aux_image_name += "@.png";
    else
      aux_image_name += ".png";

    Image& aux_image = renderer.get_aux_image();
    aux_image.save(aux_image_name);

    //Render THIRD y-image
    Settings::up_vector = Thea::Vector3(0, 1, 0);
    string cmd_line5 = "render -c aux -v ";
    cmd_line5 += cmd2;
    int status5 = renderer.exec(cmd_line5);

    if (status5 == -1)
    {
      THEA_ERROR << "Could not render the y-image for " << mesh_processor_ptr->getMeshPath();
      return false;
    }

    aux_image_name = FilePath::concat(FilePath::concat(dataset_path, DEPTH_IMAGES_FOLDER), FilePath::baseName(mesh_processor_ptr->getMeshPath()) + format("_dep_%06ld_%06ld_%06ld_%06ld", render_index, point_index, cam_dist_index + camera_starting_index, upv_index));
    if (mesh_processor_ptr->isUsedForValidation())
      aux_image_name += "@.png";
    else
      aux_image_name += ".png";

    aux_image = renderer.get_aux_image();
    aux_image.save(aux_image_name);

    Settings::up_vector = tmp;
  }
  // SDF change starts
  //Render FIFTH the SDF image
  //string cmd_line5 = "render -c sdf -v ";
  //cmd_line5 += cmd2;
  //int status5 = renderer.exec(cmd_line5);

  //if (status5 == -1)
  //{
  //  THEA_ERROR << "Could not render the sdf image for " << mesh_processor_ptr->getMeshPath();
  //  return false;
  //}

  //std::string sdf_image_name = FilePath::concat(FilePath::concat(dataset_path, AUX_IMAGES_FOLDER), FilePath::baseName(mesh_processor_ptr->getMeshPath()) + format("_aux_%06ld_%06ld_%06ld_%06ld", render_index, point_index, cam_dist_index + camera_starting_index, upv_index));
  //if (mesh_processor_ptr->isUsedForValidation())
  //  sdf_image_name += "@.png";
  //else
  //  sdf_image_name += ".png";

  //Image& sdf_image = renderer.get_sdf_image();
  //sdf_image.save(sdf_image_name);
  // SDF change ends

  // SDF/UP change starts
  //Render FIFTH the AUX image
  Thea::Vector3 tmp = Settings::up_vector;
  if (Settings::use_consistent_coord)
  {
    Settings::up_vector = Thea::Vector3(0, 0, 1);
  }

  string cmd_line5 = "render -c aux -v ";
  cmd_line5 += cmd2;
  int status5 = renderer.exec(cmd_line5);

  if (status5 == -1)
  {
    THEA_ERROR << "Could not render the aux image for " << mesh_processor_ptr->getMeshPath();
    return false;
  }

  std::string aux_image_name = FilePath::concat(FilePath::concat(dataset_path, AUX_IMAGES_FOLDER), FilePath::baseName(mesh_processor_ptr->getMeshPath()) + format("_aux_%06ld_%06ld_%06ld_%06ld", render_index, point_index, cam_dist_index + camera_starting_index, upv_index));
  if (mesh_processor_ptr->isUsedForValidation())
    aux_image_name += "@.png";
  else
    aux_image_name += ".png";

  Image& aux_image = renderer.get_aux_image();
  aux_image.save(aux_image_name);

  if (Settings::use_consistent_coord)
  {
    Settings::up_vector = tmp;
  }
  // SFD/UP change ends

	return true;
}

void RenderViews::find_visible_points(TheaSet<uint32>& visible_point_indices, const Image& face_ids_image,const Vector3& view_vector)
{
	// TODO: From the original set of sampled points, find which are visible in the currently rendered image
	// need the image as input here - or edit ShapeRenderer for this
	int image_width = face_ids_image.getWidth();
	int image_height = face_ids_image.getHeight();

	//TheaSet<uint32> visible_point_set;

	int bytes_per_pixel = face_ids_image.getBitsPerPixel() / 8;
	for (int i = 0; i < image_height; ++i)
	{
		uint8 const * scanline = (uint8 const *)face_ids_image.getScanLine(i);
		for (int j = 0; j < image_width; ++j)
		{
			uint8 r = scanline[bytes_per_pixel * j + Image::Channel::RED];
			uint8 g = scanline[bytes_per_pixel * j + Image::Channel::GREEN];
			uint8 b = scanline[bytes_per_pixel * j + Image::Channel::BLUE];

			if (r == 255 && g == 255 && b == 255)
				continue; // white background, so ingore

			if (b & 0x80) // if the msb is set, it means we have a point
			{
				b = b & 0x7F; // 'AND' the blue channel with 0111 1111 to turn off the msb 
				uint32 point_id = r + g * 256 + b * 256 * 256;
				if (point_id<0 || point_id>samples_normals.size())
				{
					THEA_ERROR << "Point id is invalid (it is indexing a position outside sample points vector bounds): " << point_id;
					continue;
				}
				float normal_dot_view = samples_normals[point_id].dot(view_vector);
				// Do not count the point as visible if its normal dot view vector is smaller than the threshold
				if (normal_dot_view < Settings::point_rejection_angle)
				{
					continue;
				}
				visible_point_indices.insert(point_id);
			}
		}
	}
	//std::copy(visible_point_set.begin(), visible_point_set.end(), std::back_inserter(visible_point_indices));
}
