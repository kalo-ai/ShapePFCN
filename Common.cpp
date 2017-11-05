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

#include "Common.hpp"

string          Settings::train_meshes_path = "data/psbAirplane";
string          Settings::test_meshes_path = "data/psbAirplane";
int             Settings::pretraining_batch_size = 64;
int             Settings::pretraining_batch_splits = 2; // 2 for Tesla M40 [24GB], 4 for TitanX [12GB], 8 for 980M [8GB] (multiply by 2 in windows...)
int             Settings::training_batch_size = 64;
int             Settings::training_batch_splits = 64; // 8 for Tesla M40, won't fit in others - for CPU training (too slow), use 1
int             Settings::pretraining_num_epochs = 150;
int             Settings::training_num_epochs = 50;
string          Settings::gpu_use = "0";
bool            Settings::use_upright_coord = false; // SDF/UP CHANGE
bool            Settings::use_consistent_coord = false; // SDF/UP CHANGE
int             Settings::render_size = 512; 
int             Settings::num_sample_points  = 1024;
Thea::Vector3   Settings::up_vector = Thea::Vector3(0.0, 0.0, 1.0);
bool            Settings::do_not_use_stochastic_mvfcn = false;
bool            Settings::skip_train_rendering = false;
bool            Settings::skip_test_rendering = false;
bool            Settings::skip_training = false;
bool            Settings::skip_testing = false;
bool            Settings::skip_fcn = false;
bool            Settings::skip_fcn_train = false;
bool            Settings::skip_mvfcn = false;
bool            Settings::skip_mvfcn_train = false;
bool            Settings::skip_crf_train = false;
bool            Settings::do_not_use_crf_mvfcn = false;
bool            Settings::do_not_use_pretrained_model = false;
bool            Settings::do_only_rendering = false;
string          Settings::pooling_type = "max";
int             Settings::max_number_of_faces = 500000;
bool			Settings::baseline_rendering = false;
float			Settings::fov = 5.0f * M_PI / 180.0f;   // 5 degrees FOV
int				Settings::num_cam_distances = 2;		  // .5, 1R  (look at Common.cpp:187, check orbit distances if you change this)
int				Settings::max_images_per_distance = 20; // 3x20 = 60 max number of viewpoints x 4 rotations = 240 images max per mesh
bool			Settings::flat_shading = true;
float			Settings::point_rejection_angle = cos(M_PI / 4.0f);

istream &
operator>>(istream & in, Thea::Vector3 & v)
{
  Thea::Real x, y, z;
  in >> x >> y >> z;

  v = Thea::Vector3(x, y, z);

  return in;
}

template <typename T>
bool
parseArg(int argc, char * argv[], int index, T & value)
{
  if (index >= argc)
    return false;

  istringstream iss(argv[index]);
  if (!(iss >> value))
    return false;

  return true;
}


bool
usage(int argc, char * argv[])
{
  cout << "Usage: " << argv[0] << " [OPTIONS]\n"
    << '\n'
    << "Options:\n"
    << "  --train-meshes-path  <'string'>       : Directory with training meshes (objs, offs, 3ds, daes) and part labels. By specifying this, the program enters training mode.\n"
    << "  --test-meshes-path  <'string'>        : Directory with test meshes. Can be also a single test mesh filename.\n"
    << "  --pretraining-batch-size  <int>       : Batch size used for pretraining\n"
    << "  --training-batch-size  <int>          : Batch size used for mvfcn training\n"
    << "  --pretraining-batch-splits  <int>     : Split batches into smaller minibatches such that each split can fit in your GPU/CPU memory during pretraining\n"
    << "  --training-batch-splits  <int>        : Split batches into smaller minibatches such that each split can fit in your GPU/CPU memory during mvfcn training\n"
    << "  --pretraining-num-epochs  <int>       : Number of pre-training epochs\n"
    << "  --training-num-epochs  <int>          : Number of training epochs\n"
    << "  --gpu-use  <'string'>                 : Specify 'false' to use CPU. Specify '<id>' to use GPU with given id e.g., --gpu-use '0' uses first GPU. Specify 'all' to use all GPUs [unstable].\n"
    << "  --use-upright-coord                   : Use the upright coordinate (axis is specified with up-vector) instead x^2+y^2+z^2 as third channel. Make sure that models are consistently oriented according to y-axis!!!\n"
//    << "  --use-consisent-coord                 : Use x, y, z raw input. Make sure that models are consistently oriented according to all axes!!!\n" // experimental feature
    << "  --render-size  <int>                  : Size of rendered images to use during the rendering loop.\n"
    << "  --num-sample-points  <int>            : Number of surface sample points to use for camera placement\n"
    << "  --up-vector  \"<double double double>\"   : Upright vector (use double quotes). Default is 0 0 0 which means that an exhaustive search over camera upright orientations will be used\n"
    << "  --do-not-use-stochastic-mvfcn         : Do not use stochastic version of joint MVFCN (see paper) - this means that training can happen only in the CPU (not well tested option and too slow)\n"
    << "  --skip-train-rendering                : Skip rendering during training - assume that all images necessary to train the network are already stored in the predefined folders.\n"
    << "  --skip-test-rendering                 : Skip rendering during testing - assume that all images necessary to test the network are already stored in the predefined folders.\n"
    << "  --skip-training                       : Skip training. Useful for testing only [note: use train-meshes-path to specify path for trained model.]\n"
    << "  --skip-testing                        : Skip testing. Useful for training only / evaluate training error\n"
    << "  --skip-fcn                            : Skip fcn training and fcn inference during testing. It assumes that learned models and mesh probabilities have been saved in a previous run - only useful for debugging the CRF and MVFCN \n"
    << "  --skip-fcn-train                      : Skip fcn training only. It assumes that learned models and mesh probabilities have been saved in a previous run - only useful for debugging the CRF and MVFCN \n"
    << "  --skip-mvfcn                          : Skip mvfcn training and mvfcn inference during testing. It assumes that learned models and mesh probabilities have been saved in a previous run - only useful for debugging the CRF\n"
    << "  --skip-mvfcn-train                    : Skip mvfcn training only. It assumes that the learned CRF parameters have been saved in a previous run - only useful for debugging the CRF\n"
    << "  --skip-crf-train                      : Skip training of CRF - it assumes that the learned CRF parameters have been saved in a previous run - useful for testing the MVFCN\n"
    << "  --do-not-use-crf-mvfcn                : Do not incorporate CRF into the MVFCN during training\n"
    << "  --do-not-use-pretrained-model         : Train from scratch - do not use pretrained VGG model on images\n"
    << "  --do-only-rendering                   : Perform only rendering - useful for performing the rendering stage without any model learning/testing for a given dataset\n"
    << "  --pooling-type                        : pooling across views can be either 'max' or 'sum' - default is 'max'.\n"
    << "  --max-number-of-faces                 : set max #faces for GPU memory reasons (this is also something that does not need to be tuned in general)\n"
    << "  --baseline-rendering                  : Use baseline rendering, i.e. 20 views on a dodecahedron x 4 rotated up vectors\n"
    << "  --fov  <float>                        : Field of view (in radians) for the camera used in the renderings (default is 7 degrees)\n"
    << "  --num-cam-distances  <int>            : Number of camera distances to use for camera placement (default is 4)\n"
    << "  --max-images-per-distance  <int>      : Maximum number of images per camera distance (default is 80)\n"
    << "  --flat-shading                        : Flat shading\n"
    << "  --point-rejection-angle               : Cosine of the angle between view vector and point to reject points as non visible ( default is cos(PI/4) )\n"
    << flush;

  return true;
}


bool
parseSetting(int argc, char * argv[], int & index)
{
  string arg = argv[index];

       if (arg == "--train-meshes-path")   return parseArg(argc, argv, ++index, Settings::train_meshes_path);
  else if (arg == "--test-meshes-path")    return parseArg(argc, argv, ++index, Settings::test_meshes_path);
  else if (arg == "--pretraining-batch-size")    return parseArg(argc, argv, ++index, Settings::pretraining_batch_size);
  else if (arg == "--training-batch-size")    return parseArg(argc, argv, ++index, Settings::training_batch_size);
  else if (arg == "--pretraining-batch-splits")    return parseArg(argc, argv, ++index, Settings::pretraining_batch_splits);
  else if (arg == "--training-batch-splits")    return parseArg(argc, argv, ++index, Settings::training_batch_splits);
  else if (arg == "--pretraining-num-epochs")    return parseArg(argc, argv, ++index, Settings::pretraining_num_epochs);
  else if (arg == "--training-num-epochs")    return parseArg(argc, argv, ++index, Settings::training_num_epochs);
  else if (arg == "--gpu-use")    return parseArg(argc, argv, ++index, Settings::gpu_use);
  else if (arg == "--use-upright-coord")    { Settings::use_upright_coord = true; return true; }
  else if (arg == "--use-consistent-coord")    { Settings::use_consistent_coord = true; Settings::use_upright_coord = true;  return true; }  // consistent means also upright orientation
  else if (arg == "--render-size")    return parseArg(argc, argv, ++index, Settings::render_size);
  else if (arg == "--num-sample-points")        return parseArg(argc, argv, ++index, Settings::num_sample_points);
  else if (arg == "--up-vector")           return parseArg(argc, argv, ++index, Settings::up_vector);
  else if (arg == "--do-not-use-stochastic-mvfcn") { Settings::do_not_use_stochastic_mvfcn = true; return true; }
  else if (arg == "--skip-train-rendering") { Settings::skip_train_rendering = true; return true; }
  else if (arg == "--skip-test-rendering") { Settings::skip_test_rendering = true; return true; }
  else if (arg == "--skip-training") { Settings::skip_training = true; return true; }
  else if (arg == "--skip-testing") { Settings::skip_testing = true; return true; }
  else if (arg == "--skip-fcn") { Settings::skip_fcn = true; return true; }
  else if (arg == "--skip-fcn-train") { Settings::skip_fcn_train = true; return true; }
  else if (arg == "--skip-mvfcn") { Settings::skip_mvfcn = true; return true; }
  else if (arg == "--skip-mvfcn-train") { Settings::skip_mvfcn_train = true; return true; }
  else if (arg == "--skip-crf-train") { Settings::skip_crf_train = true; return true; }
  else if (arg == "--do-not-use-crf-mvfcn") { Settings::do_not_use_crf_mvfcn = true; return true; }
  else if (arg == "--do-not-use-pretrained-model") { Settings::do_not_use_pretrained_model = true; return true; }
  else if (arg == "--do-only-rendering") { Settings::do_only_rendering = true; return true; }
  else if (arg == "--pooling-type") { return parseArg(argc, argv, ++index, Settings::pooling_type); }
  else if (arg == "--max-number-of-faces") { return parseArg(argc, argv, ++index, Settings::max_number_of_faces); }
  else if (arg == "--baseline-rendering") { Settings::baseline_rendering = true; return true; }
  else if (arg == "--fov")        return parseArg(argc, argv, ++index, Settings::fov);
  else if (arg == "--num-cam-distances")        return parseArg(argc, argv, ++index, Settings::num_cam_distances);
  else if (arg == "--max-images-per-distances")        return parseArg(argc, argv, ++index, Settings::max_images_per_distance);
  else if (arg == "--flat-shading") { Settings::flat_shading = true; return true; }
  else if (arg == "--point-rejection-angle") { return parseArg(argc, argv, ++index, Settings::point_rejection_angle); }
  return false;
}

bool
parseSettings(int argc, char * argv[])
{
  for (int i = 1; i < argc; ++i)
  {
    string arg = argv[i];
    if (Thea::beginsWith(arg, "--"))
    {
      if (!parseSetting(argc, argv, i))
      {
        usage(argc, argv);
        return false;
      }
    }
  }

  printSettings();
  return true;
}


void
printSettings(ostream & out)
{
  out << "train-meshes-path = " << Settings::train_meshes_path << endl;
  out << "test-meshes-path = " << Settings::test_meshes_path << endl;
  out << "pretraining-batch-size = " << Settings::pretraining_batch_size << std::endl;
  out << "training-batch-size = " << Settings::training_batch_size << std::endl;
  out << "pretraining-batch-splits = " << Settings::pretraining_batch_splits << std::endl;
  out << "training-batch-splits = " << Settings::training_batch_splits << std::endl;
  out << "pretraining-num-epochs = " << Settings::pretraining_num_epochs << std::endl;
  out << "training-num-epochs = " << Settings::training_num_epochs << std::endl;
  out << "gpu-use = " << Settings::gpu_use << std::endl;
  out << "use-upright-coord = " << Settings::use_upright_coord << std::endl;
  out << "use-consistent-coord = " << Settings::use_consistent_coord << std::endl;
  out << "render-size = " << Settings::render_size << std::endl;
  out << "num-sample-points = " << Settings::num_sample_points << endl;
  out << "up-vector = " << Settings::up_vector.toString() << endl;
  out << "do-not-use-stochastic-mvfcn = " << Settings::do_not_use_stochastic_mvfcn << endl;
  out << "skip-train-rendering = " << Settings::skip_train_rendering << endl;
  out << "skip-test-rendering = " << Settings::skip_test_rendering << endl;
  out << "skip-training = " << Settings::skip_training << endl;
  out << "skip-testing = " << Settings::skip_testing << endl;
  out << "skip-fcn = " << Settings::skip_fcn << endl;
  out << "skip-fcn-train = " << Settings::skip_fcn_train << endl;
  out << "skip-mvfcn = " << Settings::skip_mvfcn << endl;
  out << "skip-mvfcn-train = " << Settings::skip_mvfcn_train << endl;
  out << "skip-crf-train = " << Settings::skip_crf_train << endl;
  out << "do-not-use-crf-mvfcn = " << Settings::do_not_use_crf_mvfcn << endl;
  out << "do-not-use-pretrained-model = " << Settings::do_not_use_pretrained_model << endl;
  out << "do-only-rendering = " << Settings::do_only_rendering << endl;
  out << "pooling-type = " << Settings::pooling_type << endl;
  out << "max-number-of-faces = " << Settings::max_number_of_faces << endl;
  out << "baseline_rendering = " << Settings::baseline_rendering << endl;
  out << "fov = " << Settings::fov << endl;
  out << "num-cam-distances = " << Settings::num_cam_distances << endl;
  out << "max-images-per-distances = " << Settings::max_images_per_distance << endl;
  out << "flat-shading = " << Settings::flat_shading << endl;
  out << "point-rejection-angle = " << Settings::point_rejection_angle << endl;
}
