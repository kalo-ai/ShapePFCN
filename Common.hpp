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

#ifndef __MVFCN_Common_hpp__
#define __MVFCN_Common_hpp__

#define GLOG_NO_ABBREVIATED_SEVERITIES
#define NO_STRICT // stupid windows conflicts
//#define SKIP_COMPILING_CAFFE_NETWORK_CODE


#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <locale>
#include <sstream>
#include <memory>
#include <iosfwd>
#include <utility>
#include <fstream>
#include <streambuf>
#include <regex>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/johnson_all_pairs_shortest.hpp>
#include <boost/regex.hpp>
#include "Vector3.hpp"
#include <Polygon3.hpp>
#include "FileSystem.hpp"
#include "FilePath.hpp"
#include "Log.hpp"
#include "Graphics/GeneralMesh.hpp"
#include "Graphics/DisplayMesh.hpp"
#include "Graphics/MeshGroup.hpp"
#include "Algorithms/BestFitSphere3.hpp"
#include "Algorithms/MeshSampler.hpp"
#include "Algorithms/MeshFeatures/Local/ShapeDiameter.hpp"
#include "Algorithms/MeshTriangles.hpp"


#define PRENDERED_IMAGES_FOLDER          "_MVFCN_PRendered_Images_"
#define DEPTH_IMAGES_FOLDER              "_MVFCN_Depth_Images_"
#define AUX_IMAGES_FOLDER                "_MVFCN_Aux_Images_" // SDF/UP change
#define SEGMENTATION_IMAGES_FOLDER       "_MVFCN_Segmentation_Images_"
#define SEGMENTATION_COLOR_IMAGE_FOLDER  "_MVFCN_Segmentation_Color_Images_"
#define TRIANGLEID_IMAGES_FOLDER         "_MVFCN_TriangleID_Images_"
#define MESH_METADATA_FOLDER             "_MVFCN_Mesh_Metadata_"     // rename to RENDERING_MESH_METADATA_FOLDER
#define LEARNING_METADATA_FOLDER         "_MVFCN_Learning_Metadata_"
#define OUTPUT_MESH_METADATA_FOLDER      "_MVFCN_Output_Mesh_Metadata_"
#define OUTPUT_SEGMENTANTIONS_FOLDER     "_MVFCN_Output_Segmentations_"
#define MESH_LIST                         "mesh_list.txt"
#define IMAGE_LIST                       "prendered_images_list.txt"
#define LABEL_LIST                       "segmentation_images_list.txt"
#define BASE_TRAIN_MODEL_FILENAME        "frontend_vgg_train_net.txt"
#define BASE_TEST_MODEL_FILENAME         "frontend_vgg_test_net.txt"
#define BASE_DEPLOY_MODEL_FILENAME       "frontend_vgg_deploy_net.txt"
#define BASE_SOLVER_FILENAME             "frontend_vgg_solver.txt"
#define MVFCN_TRAIN_MODEL_BASEFILENAME   "mvfcn_train"
#define MVFCN_TEST_MODEL_BASEFILENAME    "mvfcn_test"
#define MVFCN_SOLVER_FILENAME            "mvfcn_solver.txt"
#define PRETRAINED_MODEL_FILENAME        "vgg_conv.caffemodel"
#define OUTPUT_PRETRAINED_MODEL_FILENAME "frontend_vgg_model"
#define OUTPUT_MODEL_FILENAME            "mvfcn_model"
#define LABELS_FILENAME                  "labels.txt"
#define VALIDATION_DATA_FILENAME         "val.txt"
#define ACCURACY_FILENAME                "accuracy"
#define CRF_PARAMETERS_FILENAME          "crf_parameters.txt"
#define DISJOINT_CRF_PARAMETERS_FILENAME "disjoint_crf_parameters.txt"
#define STATE_FILENAME                   "state.txt"
#ifndef M_PI
#define M_PI 3.141592653589793238463
#endif
#define MAX_GPU_MEMORY_AVAILABLE_FOR_TRAINING         13958643712    // i.e., M40 24GB - assume 13GB because CAFFE seems to need a +10GB overhead for some reason (?)
#define GPU_MEMORY_REQUIRED_PER_VIEW                  573464576      // about half GB
#define NUM_VIEWS_MVFCN                               24L
#define GPU_MEMORY_AVAILABLE_FOR_PAIRWISE_FEATURES    (MAX_GPU_MEMORY_AVAILABLE_FOR_TRAINING - GPU_MEMORY_REQUIRED_PER_VIEW*NUM_VIEWS_MVFCN)
#define MAX_NUMBER_OF_PAIRWISE_ENTRIES                (GPU_MEMORY_AVAILABLE_FOR_PAIRWISE_FEATURES / 4L)  // divide by 4 since float=4 bytes


using namespace std;

struct IndexAttribute
{
	long index;

	IndexAttribute() : index(-1) {}
  void draw(Thea::Graphics::RenderSystem &render_system, Thea::Graphics::RenderOptions const &options) const {}
};

typedef Thea::Graphics::GeneralMesh<IndexAttribute, Thea::Graphics::NullAttribute, IndexAttribute> Mesh;
typedef Thea::Graphics::MeshGroup<Mesh> MeshContainer;

struct ReadCallback : public Thea::Graphics::MeshCodec<Mesh>::ReadCallback
{
  void vertexAdded(Mesh * mesh, long index, Thea::Graphics::IncrementalMeshBuilder<Mesh>::VertexHandle vertex)
	{
		vertex->attr().index = index;
	}

  void faceAdded(Mesh * mesh, long index, Thea::Graphics::IncrementalMeshBuilder<Mesh>::FaceHandle face)
	{
		face->attr().index = index;
	}
};


struct Settings
{
  static int                num_sample_points;   // maybe not needed
  static Thea::Vector3      up_vector;           // maybe not needed
  static int                render_size;
  static int                pretraining_num_epochs;
  static int                training_num_epochs;
  static string             train_meshes_path;
  static string             test_meshes_path;
  static int                pretraining_batch_size;
  static int                training_batch_size;
  static int                pretraining_batch_splits;
  static int                training_batch_splits;
  static string             gpu_use;
  static bool               use_upright_coord; // SDF/UP CHANGE
  static bool               use_consistent_coord; // SDF/UP CHANGE
  static bool               do_not_use_stochastic_mvfcn;
  static bool               skip_train_rendering;
  static bool               skip_test_rendering;
  static bool               skip_training;
  static bool               skip_testing;
  static bool               skip_fcn;
  static bool               skip_fcn_train;
  static bool               skip_mvfcn;
  static bool               skip_mvfcn_train;
  static bool               skip_crf_train;
  static bool               do_not_use_crf_mvfcn;
  static bool               do_not_use_pretrained_model;
  static bool               do_only_rendering;
  static string             pooling_type;
  static int                max_number_of_faces;
  static bool				baseline_rendering;
  static float				fov;
  static int				num_cam_distances;		 // needed for non-baseline rendering case
  static int				max_images_per_distance; // needed for non-baseline rendering case
  static bool				flat_shading;
  static float				point_rejection_angle;
};

istream & operator>>(istream & in, Thea::Vector3 & v);
bool parseSetting(int argc, char * argv[], int & index);
bool parseSettings(int argc, char * argv[]);
void printSettings(ostream & out = cout);
bool usage(int argc, char * argv[]);


typedef Thea::Graphics::DisplayMesh DMesh;
typedef Thea::Graphics::MeshGroup<DMesh> MG;

typedef boost::property<boost::edge_weight_t, float> EdgeProperty;
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::no_property, EdgeProperty> ProximityGraph;

enum ViewPoolingOperator { MAX_VIEW_POOLING, SUM_VIEW_POOLING };

#endif
