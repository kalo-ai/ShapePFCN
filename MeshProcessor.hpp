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

#ifndef __MVFCN_MeshProcessor_hpp__
#define __MVFCN_MeshProcessor_hpp__

#include "Common.hpp"
#include <Algorithms/MeshKDTree.hpp>  // SDF/UP change
#include <omp.h>

#ifndef SKIP_COMPILING_CAFFE_NETWORK_CODE
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif

typedef std::pair<DMesh const *, long> FaceRef;
typedef TheaUnorderedMap<FaceRef, Thea::uint32> FaceIndexMap;
typedef Thea::Algorithms::MeshKDTree<DMesh> KDTree; // SDF/UP change

struct View
{
	Thea::Vector3 dir;
	bool has_eye;
	Thea::Vector3 eye;
	Thea::Vector3 up;

	View() : dir(-1, -1, -1), has_eye(false), up(0, 1, 0) {}
};

struct MeshReadCallback : public Thea::Graphics::MeshCodec<DMesh>::ReadCallback
{
	FaceIndexMap & tri_ids;
	FaceIndexMap & quad_ids;

	MeshReadCallback(FaceIndexMap & tri_ids_, FaceIndexMap & quad_ids_) : tri_ids(tri_ids_), quad_ids(quad_ids_) {}

	void faceAdded(DMesh * mesh, long index, Thea::Graphics::IncrementalMeshBuilder<DMesh>::FaceHandle face)
	{
		if (face.hasTriangles())
		{
			long base_tri = face.getFirstTriangle();
			for (long i = 0; i < face.numTriangles(); ++i)
			{
				FaceRef ref(mesh, base_tri + i);
				tri_ids[ref] = (Thea::uint32)index;

				// THEA_CONSOLE << ref.first->getName() << ": Triangle " << ref.second << " has id " << index;
			}
		}

		if (face.hasQuads())
		{
			long base_quad = face.getFirstQuad();
			for (long i = 0; i < face.numQuads(); ++i)
			{
				FaceRef ref(mesh, base_quad + i);
				quad_ids[ref] = (Thea::uint32)index;

				// THEA_CONSOLE << ref.first->getName() << ": Quad " << ref.second << " has id " << index;
			}
		}
	}
};

struct Model
{
  Model() : kdtree(NULL) {}   // SDF/UP change
	Thea::Graphics::Camera fitCamera(Thea::Matrix4 const & transform, View const & view, Thea::Real zoom, int width, int height, double cam_distance, double mesh_radius);
	MG mesh_group;
	MG orig_mesh_group;
	TheaArray<Thea::Vector3> sample_points;
	TheaArray<Thea::Vector3> sample_normals;
	TheaArray<int> face_labels;
	FaceIndexMap tri_ids, quad_ids;
	double mesh_radius;
  double axis_length[3]; // SDF/UP change
  double min_axis_values[3]; // SDF/UP change
  KDTree * kdtree;  // SDF/UP change
};

class MeshProcessor
{
public:
  MeshProcessor(const string & mesh_path, bool update_label_map = false, bool load_rendering_mesh = true);
  MeshProcessor(const string & mesh_path, bool update_label_map, std::map<string, int>& label_map, bool load_rendering_mesh);

  inline const MeshContainer::Ptr getMeshPtr() { return mesh_container_ptr; }
  inline const string getMeshPath() { return _mesh_path; }
  vector<string> searchForImages(const string& search_path);
  inline void setValidationFlag() { use_for_validation = true; };
  inline  bool isUsedForValidation() { return use_for_validation; };
  inline float getMeshRadius() { return mesh_bsphere.getRadius(); }
  inline Thea::Vector3 getMeshCenter() { return mesh_bsphere.getCenter(); }
  inline vector<int> getGroundTruthFaceLabels() { return ground_truth_face_labels; }
  inline int getNumberOfPairwiseFeatures() { return number_of_pairwise_features; }
  inline unsigned long getNumberOfEntriesInPairwiseFeatures() { return number_of_entries_in_pairwise_features; }
  inline int getNumberOfFaces() { return number_of_faces; }
  void freeMeshData();
  Model* getModel() { return &model; }

  static string convert_raw_label_name(const string& label);

#ifndef SKIP_COMPILING_CAFFE_NETWORK_CODE
  void initFaceLabelProbabilities(const size_t num_classes, const ViewPoolingOperator& view_pooling_operator);
  void projectImageLabelProbabilitiesToMesh(const std::vector<cv::Mat>& output_channels, const cv::Mat& image_to_triangle_ids, const ViewPoolingOperator& view_pooling_operator);
  void computeMeshNormalizedUnaryFeatures(const string& input_output_filename, bool rewrite_output_filename);
  void computeMeshPairwiseFeatures(const string& input_output_filename, bool rewrite_output_filename);
  void freeMeshCRFData();
#endif

private:
  string _mesh_path;
  MeshContainer::Ptr mesh_container_ptr;
  unsigned int number_of_faces;
  bool use_for_validation;
  Thea::Ball3 mesh_bsphere;  //Ball3 is pretty light compared to BestFitSphere3
  float mesh_bsphere_radius; // needed separately because mesh_bsphere might need to be delete to save mem
  double axis_length[3], min_axis_values[3]; // UP change
  vector<int> ground_truth_face_labels;        // K faces x 1
  vector<float> face_areas;                   // K faces x 1
  int number_of_pairwise_features;
  unsigned long number_of_entries_in_pairwise_features;
  float adjacent_faces_ball_radius;

  Model model;

  void computeMeshBoundingSphere(MeshContainer & mesh, Thea::Ball3 & bsphere);
  void computeAxisLength(MeshContainer & mesh); // UP change
  void computeFaceAreas();
  void setGroundTruthLabels(std::map<string, int>& label_map, bool do_not_update_label_map);
  void writeGroundTruthLabels();
  void cloneMeshGroup(MG const & src, MG & dst);

#ifndef SKIP_COMPILING_CAFFE_NETWORK_CODE
  friend class MeshCRF;

  vector<float> face_pairwise_features_flattened;   // sequentially stores <face1 id + max_faces * feature_id>, <face2 id>, feature_value>, i.e. 3 * #CRF pairwise entries
  cv::Mat geodesic_distances;
  cv::Mat inferred_face_labels;               // K faces x 1
  cv::Mat face_unary_probabilities;           // K faces x L probabilities
  cv::Mat face_log_unary_features;       // K faces x L probabilities
  cv::Mat face_mf_probabilities;              // K faces x L probabilities, extracted by mean-field

  void computeGeodesicDistances(const vector< vector< Mesh::Face*> >& mesh_faces);
  bool outputCRFPairwiseFeatures(const string& output_filename);
  bool inputCRFPairwiseFeatures(const string& input_filename);
  bool outputCRFUnaryFeatures(const string& output_filename);
  bool inputCRFUnaryFeatures(const string& input_filename);

  bool outputMFprobs(const string& output_filename);
  bool outputMFlabels(const string& output_filename, const std::map<string, int>& label_map);
  bool computeMostLikelyMFLabels();
  float computeMeshLabelingAccuracy(const std::map<string, int>& label_map);
#endif
};


class FarthestPointGMesh
{
public:
	FarthestPointGMesh(Thea::Vector3 const & center_)
		: center(center_), max_sqdist(0) {}

	bool operator()(Mesh const & mesh)
	{
		for (Mesh::VertexConstIterator vi = mesh.verticesBegin(); vi != mesh.verticesEnd(); ++vi)
		{
			Thea::Real sqdist = (vi->getPosition() - center).squaredLength();
			if (sqdist > max_sqdist)
				max_sqdist = sqdist;
		}

		return false;
	}

	Thea::Real getFarthestDistance() const { return sqrt(max_sqdist); }

private:
	Thea::Vector3 center;
	Thea::Real max_sqdist;
};


#endif