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

#include "MeshProcessor.hpp"

Thea::Graphics::Camera Model::fitCamera(Thea::Matrix4 const & transform, View const & view, Thea::Real zoom, int width, int height, double cam_distance, double mesh_radius)
{

	// Make absolutely sure these are unit vectors
	Thea::Vector3 dir = view.dir.unit();
	Thea::Vector3 up = view.up.unit();
	Thea::Vector3 eye;

	if (view.has_eye)
	{
		eye = view.eye;
	}
	else
	{
		throw Thea::Error("Viewpoint has not eye position");
	}

	Thea::CoordinateFrame3 cframe = Thea::CoordinateFrame3::fromViewFrame(eye, eye + dir, up);

	// Projection
	static Thea::Real const HALF_WIDTH = 0.5;
	Thea::Real hw = 0, hh = 0;
	if (height > width)
	{
		Thea::Real aspect_ratio = height / (Thea::Real)width;
		hw = HALF_WIDTH;
		hh = aspect_ratio * HALF_WIDTH;
	}
	else
	{
		Thea::Real aspect_ratio = width / (Thea::Real)height;
		hw = aspect_ratio * HALF_WIDTH;
		hh = HALF_WIDTH;
	}

	Thea::Real near_dist = .01; // V: g++ warning: may be used uninitialized
	Thea::Real far_dist = 10.0;   // V: g++ warning: may be used uninitialized
	if (Settings::baseline_rendering)
	{
		near_dist = cam_distance - mesh_radius; // cam_distance is the distance from eye to mesh bsphere center
		far_dist = cam_distance + mesh_radius;
	}
	else
	{
		near_dist = cam_distance - mesh_radius; // cam_distance is the distance from eye to corresponding sample point on mesh surface
		far_dist = cam_distance + 2.0*mesh_radius;
	}

	// Set left and top according to fov and near distance
	hw = near_dist * tan(Settings::fov / 2.0f);
	hh = near_dist * tan(Settings::fov / 2.0f);

	return Thea::Graphics::Camera(cframe,
		Thea::Graphics::Camera::ProjectionType::PERSPECTIVE, -hw, hw, -hh, hh, near_dist, far_dist, Thea::Graphics::Camera::ProjectedYDirection::UP);
}


MeshProcessor::MeshProcessor(const string &mesh_path, bool update_label_map, bool load_rendering_mesh)
{
  std::map<string, int> dummy;
  MeshProcessor(mesh_path, update_label_map, dummy, load_rendering_mesh);
}

MeshProcessor::MeshProcessor(const string &mesh_path, bool update_label_map, std::map<string, int>& label_map, bool load_rendering_mesh)
{
  use_for_validation = false;
	_mesh_path = mesh_path;
  std::replace(_mesh_path.begin(), _mesh_path.end(), '\\', '/');

	try
	{
		THEA_CONSOLE << "Loading mesh: " << _mesh_path;
		ReadCallback   read_callback;

		Thea::CodecOBJ<Mesh>::Ptr codec_obj(new Thea::CodecOBJ<Mesh>(Thea::CodecOBJ<Mesh>::ReadOptions().setIgnoreNormals(false).setIgnoreTexCoords(true)));
		Thea::CodecOFF<Mesh>::Ptr codec_off(new Thea::CodecOFF<Mesh>());
		//Thea::Codec3DS<Mesh>::Ptr codec_3ds(new Thea::Codec3DS<Mesh>(Thea::Codec3DS<Mesh>::ReadOptions().setIgnoreTexCoords(true)));

		TheaArray< Thea::Graphics::MeshCodec<Mesh>::Ptr > read_codecs;
		read_codecs.push_back(codec_obj);
		read_codecs.push_back(codec_off);
		//read_codecs.push_back(codec_3ds);

		mesh_container_ptr = MeshContainer::Ptr(new MeshContainer(_mesh_path));
		mesh_container_ptr->load(_mesh_path, read_codecs, &read_callback);

		// find part metadata / bpsheres
    number_of_pairwise_features = 2; // changes this if you add more features!!!
    adjacent_faces_ball_radius = .05f; // change to .1f if euclidean distances are used
    number_of_entries_in_pairwise_features = 0;
		number_of_faces = 0;
    for (Thea::Graphics::MeshGroup<Mesh>::MeshConstIterator mi = mesh_container_ptr->meshesBegin(); mi != mesh_container_ptr->meshesEnd(); ++mi)
      number_of_faces += (**mi).numFaces();

    computeMeshBoundingSphere(*mesh_container_ptr, mesh_bsphere);
    computeAxisLength(*mesh_container_ptr); // UP CHANGE
    setGroundTruthLabels(label_map, update_label_map);
    writeGroundTruthLabels();
    computeFaceAreas();    
	}
	catch (std::exception & e__)
	{
		THEA_ERROR << "An error occurred while loading mesh: " << e__.what();
	}

	// Moved this here from ShapeRenderer
	// Load the mesh again as a DisplayMesh this time, and encapsulate it into a Model struct we can pass to each ShapeRenderer
	// (temporary hack to see if this works)
  if (!load_rendering_mesh)
    return;

	try
	{
		MeshReadCallback callback(model.tri_ids, model.quad_ids);

		model.mesh_group.load(_mesh_path, Thea::Codec_AUTO(), &callback);
		cloneMeshGroup(model.mesh_group, model.orig_mesh_group);
		model.face_labels = ground_truth_face_labels;
		model.mesh_radius = mesh_bsphere.getRadius();
    model.axis_length[0] = axis_length[0];
    model.axis_length[1] = axis_length[1];
    model.axis_length[2] = axis_length[2];
    model.min_axis_values[0] = min_axis_values[0];
    model.min_axis_values[1] = min_axis_values[1];
    model.min_axis_values[2] = min_axis_values[2];

    model.kdtree = new KDTree; // SDF/UP change
    model.kdtree->add(model.orig_mesh_group); // SDF/UP change
    model.kdtree->init(); // SDF/UP change
    THEA_CONSOLE << "Created kd-tree!"; // SDF/UP change
	}
	catch (std::exception & e__)
	{
		THEA_ERROR << "An error occurred while loading mesh: " << e__.what();
	}

}


void MeshProcessor::computeMeshBoundingSphere(MeshContainer & mesh, Thea::Ball3 & bsphere)
{
	/*bsphere.clear();

	for (Thea::Graphics::MeshGroup<Mesh>::MeshConstIterator mi = mesh.meshesBegin(); mi != mesh.meshesEnd(); ++mi)
	{
		for (Mesh::VertexConstIterator vi = (**mi).verticesBegin(); vi != (**mi).verticesEnd(); ++vi)
		{
			bsphere.addPoint(vi->getPosition());
		}
	}*/

	double sum_x = 0, sum_y = 0, sum_z = 0;
	double sum_w = 0;

	Thea::Algorithms::MeshTriangles<Mesh> tris;
	tris.add(const_cast<MeshContainer &>(*mesh_container_ptr));

	Thea::Algorithms::MeshTriangles<Mesh>::TriangleArray const & tri_array = tris.getTriangles();
	for (int i = 0; i < tri_array.size(); ++i)
	{
		Thea::Vector3 c = tri_array[i].getCentroid();
		Thea::Real area = tri_array[i].getArea();

		sum_x += (area * c[0]);
		sum_y += (area * c[1]);
		sum_z += (area * c[2]);

		sum_w += area;
	}

	Thea::Vector3 center(0, 0, 0);
	if (sum_w > 0)
	{
		center[0] = (Thea::Real)(sum_x / sum_w);
		center[1] = (Thea::Real)(sum_y / sum_w);
		center[2] = (Thea::Real)(sum_z / sum_w);
	}

	Thea::Real radius = 0;

	FarthestPointGMesh fp(center);
	mesh_container_ptr->forEachMeshUntil(&fp);
	radius = fp.getFarthestDistance();

	bsphere.setCenter(center);
	bsphere.setRadius(radius);

	mesh_bsphere_radius = bsphere.getRadius();

}


// UP CHANGE
void MeshProcessor::computeAxisLength(MeshContainer & mesh)
{
  double max_x = -DBL_MAX, min_x = DBL_MAX, max_y = -DBL_MAX, min_y = DBL_MAX, max_z = -DBL_MAX, min_z = DBL_MAX;
  Thea::Algorithms::MeshTriangles<Mesh> tris;
  tris.add(const_cast<MeshContainer &>(*mesh_container_ptr));

  Thea::Algorithms::MeshTriangles<Mesh>::TriangleArray const & tri_array = tris.getTriangles();
  for (int i = 0; i < tri_array.size(); ++i)
  {
    Thea::Vector3 c = tri_array[i].getCentroid();
    max_x = max<double>(max_x, c.x());
    min_x = min<double>(min_x, c.x());
    max_y = max<double>(max_y, c.y());
    min_y = min<double>(min_y, c.y());
    max_z = max<double>(max_z, c.z());
    min_z = min<double>(min_z, c.z());
  }

  min_axis_values[0] = min_x;
  min_axis_values[1] = min_y;
  min_axis_values[2] = min_z;
  axis_length[0] = abs(max_x - min_x);
  axis_length[1] = abs(max_y - min_y);
  axis_length[2] = abs(max_z - min_z);

  string more_mesh_metadata_filename = _mesh_path + "_axis.txt";
  ofstream more_mesh_metadata_file(more_mesh_metadata_filename.c_str());
  more_mesh_metadata_file << min_x << " " << min_y << " " << min_z << " " << max_x << " " << max_y << " " << max_z << std::endl;
  more_mesh_metadata_file << axis_length[0] << " " << axis_length[1] << " " << axis_length[2] << std::endl;
  more_mesh_metadata_file.close();
}

void MeshProcessor::cloneMeshGroup(MG const & src, MG & dst)
{
	dst.clear();
	dst.setName(src.getName());

	for (MG::MeshIterator mi = src.meshesBegin(); mi != src.meshesEnd(); ++mi)
	{
		DMesh::Ptr tgt(new DMesh(**mi));
		dst.addMesh(tgt);
	}

	for (MG::GroupIterator ci = src.childrenBegin(); ci != src.childrenEnd(); ++ci)
	{
		MG::Ptr tgt(new MG(**ci));
		dst.addChild(tgt);
		cloneMeshGroup(**ci, *tgt);
	}
}

vector<string> MeshProcessor::searchForImages(const string& search_path)
{
	const boost::regex pattern(Thea::FilePath::baseName(getMeshPath()) + "_.*\\.png");

	vector<string> matching_image_filenames;

	boost::filesystem::directory_iterator end_itr; // Default ctor yields past-the-end
	for (boost::filesystem::directory_iterator i(search_path); i != end_itr; ++i)
	{
		// Skip if not a file
		if (!boost::filesystem::is_regular_file(i->status())) continue;

		// Skip if no match for V3:
		if (!boost::regex_match(i->path().filename().string(), pattern)) continue;

		// File matches, store it
		matching_image_filenames.push_back(Thea::FilePath::concat(search_path, i->path().filename().string()));
	}

	return matching_image_filenames;
}


void MeshProcessor::setGroundTruthLabels(std::map<string, int>& label_map, bool update_label_map)
{
  THEA_CONSOLE << "Setting ground truth labels for the faces of the mesh " << getMeshPath();
  ground_truth_face_labels.resize(number_of_faces, -1);

  if (!mesh_container_ptr)
  {
    THEA_ERROR << "Cannot compute ground-truth labels because mesh data were deleted!";
    return;
  }

  // MESHES WITHOUT GROUPS (e.g. OFFs, need labels txt in this case)
  if (mesh_container_ptr->numMeshes() == 1)
  {
    string labels_txt_filename = Thea::FilePath::concat(Thea::FilePath::parent(getMeshPath()), Thea::FilePath::baseName(getMeshPath()) + "_labels.txt");
    THEA_CONSOLE << "Attempting to import ground truth labels from  " << labels_txt_filename;
    ifstream labels_txt_file(labels_txt_filename);
    if (!labels_txt_file.good())
    {
      THEA_WARNING << "Could not import labels from " << labels_txt_filename << ". Will not be able to evaluate labeling accuracy";
      return;
    }

    int current_label_id = -1;
    while (!labels_txt_file.eof())
    {
      string token;
      labels_txt_file >> token;
      if (token.empty())
        continue;
      bool is_number = std::find_if(token.begin(), token.end(), [](char c) { return !std::isdigit(c); }) == token.end();
      if (is_number)
      {
        int face_id = std::stoi(token) - 1;  // BE CAREFUL: INDICES OF THE LABELS.TXT FORMAT START WITH 1!!!
        ground_truth_face_labels[face_id] = current_label_id;
      }
      else
      {
        string label_name = convert_raw_label_name(token);

        std::map<string, int>::const_iterator it = label_map.find(label_name);
        if (it != label_map.end())
          current_label_id = label_map.at(label_name);
        else
        {
          if (update_label_map)
          {
            int id = (int)label_map.size();
            THEA_CONSOLE << "Adding label " << label_name << " [id = " << id << "] to the list of labels ";
            label_map[label_name] = id;
            current_label_id = id;
          }
          else
            THEA_WARNING << "Label " << label_name << " does not exist in the training labels map (test labels != training labels)";
        }

      }
    }

    labels_txt_file.close();
    return;
  }

  // MESHES WITH GROUPS
  for (Thea::Graphics::MeshGroup<Mesh>::MeshConstIterator mi = mesh_container_ptr->meshesBegin(); mi != mesh_container_ptr->meshesEnd(); ++mi)
  {
    string label_name = convert_raw_label_name((**mi).getName());
    THEA_CONSOLE << "Found part in structured mesh with label: " << label_name;

    std::map<string, int>::const_iterator it = label_map.find(label_name);
    int label_id = -1;
    if (it != label_map.end())
      label_id = label_map.at(label_name);
    else
    {
      if (update_label_map)
      {
        int id = (int)label_map.size();
        THEA_CONSOLE << "Adding label " << label_name << " [id = " << id << "] to the list of labels ";
        label_map[label_name] = id;
        label_id = id;
      }
      else
        THEA_WARNING << "Label " << label_name << " does not exist in the training labels map (test labels != training labels)";
    }

    for (Mesh::FaceIterator fi = (**mi).facesBegin(); fi != (**mi).facesEnd(); ++fi)
    {
      ground_truth_face_labels[fi->attr().index] = label_id;
    }
  }
}

void MeshProcessor::writeGroundTruthLabels()
{
  string seg_filename = Thea::FilePath::changeCompleteExtension(_mesh_path, "seg");
  THEA_CONSOLE << "Writing ground truth face labels to " << seg_filename;
  ofstream ground_truth_labels_filename(seg_filename);
  for (unsigned int f = 0; f < number_of_faces; f++)
  {
    ground_truth_labels_filename << ground_truth_face_labels[f] << std::endl;
  }
  ground_truth_labels_filename.close();
}

void MeshProcessor::computeFaceAreas()
{
  face_areas.resize(number_of_faces, 0.0f);
  float total_face_area = 1e-12f;

  for (Thea::Graphics::MeshGroup<Mesh>::MeshConstIterator mi = mesh_container_ptr->meshesBegin(); mi != mesh_container_ptr->meshesEnd(); ++mi)
  {
    for (Mesh::FaceIterator fi = (**mi).facesBegin(); fi != (**mi).facesEnd(); ++fi)
    {
      float face_area = Thea::Polygon3::computeArea(fi->verticesBegin(), fi->verticesEnd());
      face_areas[fi->attr().index] = face_area;
      total_face_area += face_area;
    }
  }

  for (unsigned int f = 0; f < number_of_faces; ++f)
  {
    face_areas[f] /= total_face_area;
  }
}

string MeshProcessor::convert_raw_label_name(const string& label)
{
  string converted_label = label;
  converted_label.erase(std::remove_if(converted_label.begin(), converted_label.end(), [](char x){ return !( (std::isalpha(x)) | (x == '_') ); }), converted_label.end());
  std::transform(converted_label.begin(), converted_label.end(), converted_label.begin(), ::tolower);

  if (converted_label.empty())
    converted_label = "NONAME";

  if (converted_label[ converted_label.size()-1 ] == '_')
    converted_label = converted_label.substr(0, converted_label.size() - 1);;

  return converted_label;
}



void MeshProcessor::freeMeshData()
{
  THEA_CONSOLE << "Cleaning up...";
	mesh_container_ptr = NULL;
  model.sample_points.clear(); // we need to delete these, otherwise we run out of memory after rendering 10 meshes!
  model.sample_normals.clear(); // we need to delete these, otherwise we run out of memory after rendering 10 meshes!
  model.tri_ids.clear();
  model.quad_ids.clear();
  model.face_labels.clear();
  model.mesh_group.clear();
  model.orig_mesh_group.clear();
  if (model.kdtree != NULL) delete model.kdtree; // SDF/UP change
  model.kdtree = NULL; // SDF/UP change
#ifndef SKIP_COMPILING_CAFFE_NETWORK_CODE
	face_pairwise_features_flattened.clear();
  geodesic_distances.release();
#endif
}




#ifndef SKIP_COMPILING_CAFFE_NETWORK_CODE
void MeshProcessor::computeMeshPairwiseFeatures(const string& input_output_filename, bool rewrite_output_filename)
{
	// NOTE: this function loops over faces within each submesh and connects faces within each submesh (and within certain radius)
	// this will benefit us a lot if submeshes (components) correspond to meaningful parts already
	// (for PSB/COSEG testing, test meshes should have their submesh information removed)

	// feature 1: geodesic distance differences (somewhat dense)
	// feature 2: unsigned dihedral angle (very sparse)


	// attempt to load pre-existing pairwise features
  if (!rewrite_output_filename)
	  if (inputCRFPairwiseFeatures(input_output_filename))
  		return;

	// if not successful, produce them
  bool reexecute_with_smaller_ball_radius = false;
  face_pairwise_features_flattened.clear();
  face_pairwise_features_flattened.reserve(MAX_NUMBER_OF_PAIRWISE_ENTRIES);
	int number_of_faces_processed = 0;

  // Store all faces in a vector (for OpenMP reasons)
  vector< vector< Mesh::Face*> > mesh_faces(mesh_container_ptr->numMeshes());
  int part_id = 0;
  for (Thea::Graphics::MeshGroup<Mesh>::MeshConstIterator mi = mesh_container_ptr->meshesBegin(); mi != mesh_container_ptr->meshesEnd(); ++mi, ++part_id)
  {
    mesh_faces[part_id].reserve(number_of_faces);
    Mesh & mesh = **mi;
    for (Mesh::FaceIterator fi = mesh.facesBegin(); fi != mesh.facesEnd(); ++fi)
      mesh_faces[part_id].push_back(&(*fi));
  }

  THEA_LOG << "Computing pairwise geometric features...";
  THEA_CONSOLE << "Max #entries for pairwise features (due to GPU mem limitations): " << MAX_NUMBER_OF_PAIRWISE_ENTRIES;

  // precompute geodesic distances
  if (number_of_faces < 50000)  // use euclidean distances for big meshes (change to 0 to use euclidean distances always)
    computeGeodesicDistances(mesh_faces); 
    
  
  // NOTE: this code takes advantage of any mesh structure: if a mesh has components (submeshes),
  // faces between these submeshes will not be connected in the CRF. This helps individual components
  // to have consistent labels. When testing in a benchmark with manifolds (e.g., PSB), the test meshes
  // should not contain components (e.g., OBJ with groups), otherwise comparisons with other methods would
  // be unfair!
  for (int part_id = 0; part_id < (int)mesh_faces.size(); ++part_id)
  {
#pragma omp parallel for
    for (int fi = 0; fi < (int)mesh_faces[part_id].size(); ++fi)
    {
      Mesh::Face *fi_ptr = mesh_faces[part_id][fi];

      if (adjacent_faces_ball_radius > .001f) // if function repeated too many times decreasing 'adjacent_faces_ball_radius', then there must be some problem with the mesh. In this case ignore these features.
      {
        for (int fj = 0; fj < (int)mesh_faces[part_id].size(); ++fj)
        {
          if (fi == fj)
            continue;
          Mesh::Face *fj_ptr = mesh_faces[part_id][fj];

          // first feature: geodesic distance
          float sqdist = FLT_MAX;
          if (number_of_faces < 50000) // (change to 0 to use euclidean distances always)
          {
            float dist = geodesic_distances.at<float>(fi_ptr->attr().index, fj_ptr->attr().index); // if geod distances are used
            sqdist = dist*dist;  // if geod distances are used (normalized already)
          }
          else
          {
            sqdist = (fi_ptr->centroid() - fj_ptr->centroid()).squaredLength() / (mesh_bsphere_radius*mesh_bsphere_radius); 
          }
          if (sqdist > adjacent_faces_ball_radius*adjacent_faces_ball_radius) // skip if distance is larger than ball radius
            continue;

          if (face_pairwise_features_flattened.size() > MAX_NUMBER_OF_PAIRWISE_ENTRIES) // one of the threads encounters too many pairs
          {
            reexecute_with_smaller_ball_radius = true;
            break;
          }

          float id1 = float(fi_ptr->attr().index);
          float id2 = float(fj_ptr->attr().index);

#pragma omp critical // slow
          {
            face_pairwise_features_flattened.push_back(id1);   // note: this can store up to 16777216
            face_pairwise_features_flattened.push_back(id2);
            face_pairwise_features_flattened.push_back(sqdist);
          }
        }
      }

      // second feature: *** NOT SIGNED *** dihedral angle (0...pi)
      for (Mesh::Face::EdgeIterator fei = fi_ptr->edgesBegin(); fei != fi_ptr->edgesEnd(); ++fei)
      {
        Mesh::Edge * edge = *fei;
        for (Mesh::Edge::FaceIterator efi = edge->facesBegin(); efi != edge->facesEnd(); ++efi)
        {
          Mesh::Face *fj_ptr = *efi;
          if (fi_ptr->attr().index == fj_ptr->attr().index)
            continue;

          float cosphi = fi_ptr->getNormal().dot(fj_ptr->getNormal());
          float normsinphi = (fi_ptr->getNormal().cross(fj_ptr->getNormal())).length();
          float dihedral_angle = atan2(normsinphi, cosphi);
          float dissimilarity = (dihedral_angle*dihedral_angle) / (M_PI*M_PI);

          float id1 = float(Settings::max_number_of_faces + fi_ptr->attr().index); // note: no more than 16777216
          float id2 = float(fj_ptr->attr().index );

#pragma omp critical // slow
          {
            face_pairwise_features_flattened.push_back(id1); // note: this can store up to 16777216
            face_pairwise_features_flattened.push_back(id2);
            face_pairwise_features_flattened.push_back(dissimilarity);
          }
        }
      }

      // more features: CHANGE number_of_pairwise_features in constructor!

      if (omp_get_num_threads() == 1)
      {
        number_of_faces_processed++;
        if (number_of_faces_processed % (number_of_faces / 10) == 0)
          std::cout << (int)round(100.0f * (float)(number_of_faces_processed) / (float)number_of_faces) << "% ... ";
        if (number_of_faces_processed == number_of_faces)
          THEA_CONSOLE << "Done.";
      }
    } // end of iteration over faces (end of parallelism)
  } // end of loop over parts

  number_of_entries_in_pairwise_features = (unsigned long)face_pairwise_features_flattened.size();
  if (reexecute_with_smaller_ball_radius || (number_of_entries_in_pairwise_features >  MAX_NUMBER_OF_PAIRWISE_ENTRIES))
  {
    THEA_CONSOLE << "Too many pairwise feature entries (will have trouble later with fitting caffe blobs in the GPU mem). Re-executing computation with smaller ball radius for gathering adjacent faces.";
    face_pairwise_features_flattened.clear();
    face_pairwise_features_flattened.shrink_to_fit(); // in case multiple reserves have weird behavior
    mesh_faces.clear();
    adjacent_faces_ball_radius /= 2.0f;
    computeMeshPairwiseFeatures(input_output_filename, rewrite_output_filename); // recursive call that hopefully will finish
  }

	if (!input_output_filename.empty())
		if (!outputCRFPairwiseFeatures(input_output_filename))
		{
			THEA_WARNING << "Pairwise features were not written in the disk - this will cause a crash since pairwise features are too expensive to keep in memory and need to be deleted/reloaded periodically!!";
		}

  face_pairwise_features_flattened.shrink_to_fit();
}


void MeshProcessor::computeGeodesicDistances(const vector< vector< Mesh::Face*> >& mesh_faces)
{
  if (geodesic_distances.data)
    return;
  THEA_CONSOLE << "Computing geodesic distances on the mesh.";

  ProximityGraph mesh_graph(number_of_faces);
  float** geodesic_distances_f = new float*[number_of_faces];

  for (int fi = 0; fi < number_of_faces; ++fi)
  {
    geodesic_distances_f[fi] = new float[number_of_faces];
  }

  for (int part_id = 0; part_id < (int)mesh_faces.size(); ++part_id)
  {
#pragma omp parallel for
    for (int fi = 0; fi < (int)mesh_faces[part_id].size(); ++fi)
    {
      Mesh::Face *fi_ptr = mesh_faces[part_id][fi];

      // add adjacent faces
      for (Mesh::Face::EdgeIterator fei = fi_ptr->edgesBegin(); fei != fi_ptr->edgesEnd(); ++fei)
      {
        Mesh::Edge * edge = *fei;
        for (Mesh::Edge::FaceIterator efi = edge->facesBegin(); efi != edge->facesEnd(); ++efi)
        {
          Mesh::Face *fj_ptr = *efi;
          if (fi_ptr->attr().index == fj_ptr->attr().index)
            continue;

          float dist = (fi_ptr->centroid() - fj_ptr->centroid()).length();
#pragma omp critical // slow
          {
            boost::add_edge(fi_ptr->attr().index, fj_ptr->attr().index, dist, mesh_graph);
          }
        }
      }
    }
  }
  std::cout << "Created graph of " << number_of_faces << " face centers" << std::endl;
  boost::johnson_all_pairs_shortest_paths(mesh_graph, geodesic_distances_f);

  geodesic_distances = cv::Mat(number_of_faces, number_of_faces, CV_32F);
  float max_geodesic_distance = 1e-30f;
  for (int fi = 0; fi < number_of_faces; ++fi)
  {
    for (int fj = 0; fj < number_of_faces; ++fj)
    {
      if (geodesic_distances_f[fi][fj] < FLT_MAX)
      {
        max_geodesic_distance = std::max(max_geodesic_distance, geodesic_distances_f[fi][fj]);
        geodesic_distances.at<float>(fi, fj) = geodesic_distances_f[fi][fj];
      }
      else
        geodesic_distances.at<float>(fi, fj) = FLT_MAX;
    }

    delete[] geodesic_distances_f[fi];
  }
  delete[] geodesic_distances_f;

  geodesic_distances /= max_geodesic_distance;
  std::cout << "Geodesic distances are computed." << std::endl;
}



bool MeshProcessor::outputCRFPairwiseFeatures(const string& output_filename)
{
  ofstream output_file;
  output_file.open(output_filename, ios::out | ios::binary);
  if (!output_file.good())
  {
    THEA_ERROR << "Cannot write pairwise features to " << output_filename;
    output_file.close();
    return false;
  }
  output_file.write(reinterpret_cast<char*>(&face_pairwise_features_flattened[0]), face_pairwise_features_flattened.size()*sizeof(float));
  output_file.close();

  return true;
}

bool MeshProcessor::inputCRFPairwiseFeatures(const string& input_filename)
{
  face_pairwise_features_flattened.clear();
  std::ifstream input_file;
  input_file.open(input_filename, ios::in | ios::binary);
  if (!input_file.good())
  {
    input_file.close();
    return false;
  }
  input_file.seekg(0, ios::end);
  number_of_entries_in_pairwise_features = input_file.tellg() / sizeof(float);    // tellg: 2147483648 (probably?) max entries
  input_file.seekg(0, ios::beg);
  face_pairwise_features_flattened.resize(number_of_entries_in_pairwise_features);
  input_file.read(reinterpret_cast<char*>(&face_pairwise_features_flattened[0]), number_of_entries_in_pairwise_features*sizeof(float));
  input_file.close();
  THEA_CONSOLE << "Imported CRF pairwise features.";

  return true;
}


void MeshProcessor::initFaceLabelProbabilities(const size_t num_classes, const ViewPoolingOperator& view_pooling_operator)
{
  THEA_CONSOLE << "Initializing unary term probabilities (#faces=" << number_of_faces << ", #classes=" << num_classes << ")";
  face_log_unary_features = cv::Mat(number_of_faces, (int)num_classes, CV_32F);
  face_unary_probabilities = cv::Mat(number_of_faces, (int)num_classes, CV_32F);
  if (view_pooling_operator == MAX_VIEW_POOLING)
  {
    THEA_CONSOLE << "Will use max pooling.";
    face_log_unary_features = -FLT_MAX;
  }
  else
  {
    THEA_CONSOLE << "Will use sum pooling.";
    face_log_unary_features = 0.0f;
  }

  face_unary_probabilities = 1.0f / (float)num_classes;
}

void MeshProcessor::projectImageLabelProbabilitiesToMesh(const std::vector<cv::Mat>& output_channels, const cv::Mat& image_to_triangle_ids, const ViewPoolingOperator& view_pooling_operator)
{
	for (int c = 0; c < output_channels.size(); ++c)
	{
		if ((output_channels[c].size() != image_to_triangle_ids.size()))
		{
			THEA_ERROR << "Cannot project image back to mesh because output image from the net and triangle ID image have inconsistent size: " << output_channels[c].size() << " != " << image_to_triangle_ids.size();
			return;
		}

		for (int i = 0; i < output_channels[c].rows; ++i)
		{
			const float* Oi = output_channels[c].ptr<float>(i);
			const cv::Vec3b* Ti = image_to_triangle_ids.ptr<cv::Vec3b>(i);
			for (int j = 0; j < output_channels[c].cols; ++j)
			{
				unsigned int b = (unsigned int)Ti[j][0];  // BGR format
				unsigned int g = (unsigned int)Ti[j][1];
				unsigned int r = (unsigned int)Ti[j][2];
				if (r == 255 && g == 255 && b == 255) // no pixel->triangle association
					continue;
				unsigned int face_index = r + 256 * g + 65536 * b;
				if (face_index >= number_of_faces)
				{
					THEA_ERROR << "Triangle ID image has face indices larger than the size of the mesh!!! " << face_index << " >= " << number_of_faces;
					return;
				}

				if (view_pooling_operator == MAX_VIEW_POOLING)
          face_log_unary_features.at<float>(face_index, c) = max(face_log_unary_features.at<float>(face_index, c), Oi[j]);
				else if (view_pooling_operator == SUM_VIEW_POOLING)
          face_log_unary_features.at<float>(face_index, c) += Oi[j];
				else
				{
					if (c == 0)
            face_log_unary_features.at<float>(face_index, c) = 1.0f; // visible or not (only for debugging) [stored in the first label]
					else
            face_log_unary_features.at<float>(face_index, c) = 0.0f;
				}
			}
		}
	}
}


void MeshProcessor::computeMeshNormalizedUnaryFeatures(const string& input_output_filename, bool rewrite_output_filename)
{
	// // attempt to load unary features
  if (!rewrite_output_filename)
	  if (inputCRFUnaryFeatures(input_output_filename))
  		return;

  cv::Mat row_max(face_log_unary_features.rows, 1, CV_32F);
  cv::Mat row_sum(face_log_unary_features.rows, 1, CV_32F);

  cv::reduce(face_log_unary_features, row_max, 1, CV_REDUCE_MAX);
  for (int i = 0; i < face_log_unary_features.rows; ++i)
    face_log_unary_features.row(i) -= row_max.at<float>(i);
  cv::exp(face_log_unary_features, face_unary_probabilities);

	cv::reduce(face_unary_probabilities, row_sum, 1, CV_REDUCE_SUM);
	for (int i = 0; i < face_unary_probabilities.rows; ++i)
		face_unary_probabilities.row(i) /= row_sum.at<float>(i);

  //cv::Mat thr_unary_probabilities = face_unary_probabilities.clone();
  //thr_unary_probabilities.setTo(.00001f, thr_unary_probabilities < .00001f); // avoid numerical problems, overconfident unary features
  //thr_unary_probabilities.setTo(.99999f, thr_unary_probabilities > .99999f);
  //cv::log(thr_unary_probabilities, face_log_unary_features);

	if (!input_output_filename.empty())
		if (!outputCRFUnaryFeatures(input_output_filename))
		{
			THEA_ERROR << "Unary features were not written in the disk!";
		}
}

bool MeshProcessor::outputCRFUnaryFeatures(const string& output_filename)
{
	try
	{
		cv::FileStorage crf_feature_file(output_filename, cv::FileStorage::WRITE);
		crf_feature_file << "face_unary_probabilities" << face_unary_probabilities;
    crf_feature_file << "face_log_unary_features" << face_log_unary_features;
		crf_feature_file.release();
	}
	catch (cv::Exception& e__)
	{
		THEA_ERROR << e__.what();
		return false;
	}
	return true;

	///////// note - for debugging in matlab....
	//////X = load('XXX.txt'); % remove openCV metadata manually first, use face_unary_probabilities data only,  append numbers so that #columns is same
	//////X = X';
	//////X = X(1:numfaces*numclasses);
	//////X = reshape(X, numclasses, numfaces); % enter the numbers
	//////plotMeshSegmentation(mesh, X(1, :))
}




bool MeshProcessor::inputCRFUnaryFeatures(const string& input_filename)
{
  cv::Mat face_unary_probabilities_tmp, face_log_unary_features_tmp;

	try
	{
		cv::FileStorage crf_feature_file(input_filename, cv::FileStorage::READ);
    crf_feature_file["face_unary_probabilities"] >> face_unary_probabilities_tmp;
    crf_feature_file["face_log_unary_features"] >> face_log_unary_features_tmp;
		crf_feature_file.release();
    if (!face_unary_probabilities_tmp.data || !face_log_unary_features_tmp.data)
			return false;
    if ( (face_unary_probabilities_tmp.rows == 0) || (face_unary_probabilities_tmp.cols == 0)
      || (face_log_unary_features_tmp.rows == 0) || (face_log_unary_features_tmp.cols == 0) )
      return false;
	}
	catch (cv::Exception& e__)
	{
		THEA_ERROR << e__.what();
		return false;
	}
  face_unary_probabilities = face_unary_probabilities_tmp.clone();
  face_log_unary_features = face_log_unary_features_tmp.clone();
  THEA_CONSOLE << "Imported CRF unary features.";
	return true;
}



bool MeshProcessor::outputMFprobs(const string& output_filename)
{
	if (!face_mf_probabilities.data)
	{
		THEA_ERROR << "outputMFprobs() was called before mean-field is executed, or mean-field never executed properly!";
		return false;
	}

	ofstream output_file(output_filename);
	if (!output_file.good())
	{
		THEA_ERROR << "Cannot write mean field output to file: " << output_filename;
		return false;
	}

	for (int i = 0; i < face_mf_probabilities.rows; ++i)
	{
		for (int c = 0; c < face_mf_probabilities.cols; ++c)
		{
			output_file << face_mf_probabilities.at<float>(i, c) << " ";
		}
		output_file << std::endl;
	}
	output_file.close();

	return true;
}

bool MeshProcessor::outputMFlabels(const string& output_filename, const std::map<string, int>& label_map)
{
	if (!computeMostLikelyMFLabels())
	{
		THEA_ERROR << "Cannot write mean field output labels to file since mean field never ran (?)";
		return false;
	}

	ofstream output_file(output_filename);
	if (!output_file.good())
	{
		THEA_ERROR << "Cannot write mean field output labels to file: " << output_filename;
		return false;
	}

	for (unsigned int i = 0; i < number_of_faces; ++i)
	{
		output_file << inferred_face_labels.at<int>(i) << std::endl;
	}
	output_file.close();

  string lab_filename = Thea::FilePath::changeCompleteExtension(output_filename, "lab");
  ofstream output_file2(lab_filename);
  if (!output_file2.good())
  {
    THEA_ERROR << "Cannot write mean field output labels to file: " << lab_filename;
    return false;
  }
  vector < vector < int > > faces_per_label(label_map.size());
  for (unsigned int i = 0; i < number_of_faces; ++i)
  {
    faces_per_label[inferred_face_labels.at<int>(i)].push_back( i+1 ); // lab format indices start from 1
  }
  for (std::map<string, int>::const_iterator it = label_map.begin(); it != label_map.end(); ++it)
  {
    int label_id = label_map.at(it->first);
    if (faces_per_label[label_id].empty())
      continue;

    output_file2 << it->first << std::endl;
    for (unsigned int j = 0; j < faces_per_label[label_id].size(); ++j)
      output_file2 << faces_per_label[label_id][j] << " ";
    output_file2 << std::endl;
  }
  output_file2.close();

	return true;
}


bool MeshProcessor::computeMostLikelyMFLabels()
{
	if (!face_mf_probabilities.data)
	{
		THEA_ERROR << "computeMostLikelyMFLabels() was called before mean-field is executed, or mean-field never executed properly!";
		return false;
	}

	if (number_of_faces != face_mf_probabilities.rows)
	{
		THEA_ERROR << "mean field probabilities are supposed to be computed per triangle - somehow #probability measurements != number of faces (internal error)!";
		return false;
	}

	inferred_face_labels = cv::Mat(face_mf_probabilities.rows, 1, CV_32S);

	for (int i = 0; i < face_mf_probabilities.rows; ++i)
	{
		float max_prob = 0.0f;
		for (int c = 0; c < face_mf_probabilities.cols; ++c)
		{
			if (face_mf_probabilities.at<float>(i, c) > max_prob)
			{
				max_prob = face_mf_probabilities.at<float>(i, c);
				inferred_face_labels.at<int>(i) = c;
			}
		}
	}

	return true;
}


float MeshProcessor::computeMeshLabelingAccuracy(const std::map<string, int>& label_map)
{
	if (!computeMostLikelyMFLabels())
	{
		THEA_ERROR << "computeMeshLabelingAccuracy() was called before mean-field is executed!";
		return -1.0f;
	}

  if (ground_truth_face_labels.empty())
    THEA_ERROR << "Internal error: ground truth labels non-existent. They were supposed to be populated in the MeshProcessor constructor!";

  if (face_areas.empty())
    THEA_ERROR << "Internal error: face areas non-existent. They were supposed to be populated in the MeshProcessor constructor!";

	float accuracy = 0.0f;
  for (unsigned int f = 0; f < number_of_faces; ++f)
    if (inferred_face_labels.at<int>(f) == ground_truth_face_labels[f])
      accuracy += face_areas[f];

  // no need to divide with total face area - face areas are already normalized
	return accuracy;
}


void MeshProcessor::freeMeshCRFData()
{
	face_unary_probabilities.release();
  face_log_unary_features.release();
	face_mf_probabilities.release();
  face_pairwise_features_flattened.clear();
}
#endif
