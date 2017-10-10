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

#include "FCNShapes.hpp"

MVFCN::MVFCN(ViewPoolingOperator _view_pooling_type)
{
  google_logging_is_initialized = false;
  view_pooling_type = _view_pooling_type;

  if (Settings::baseline_rendering)
    num_camera_orbits = 1;
  else
    num_camera_orbits = Settings::num_cam_distances + 1; // +1 dodecahedron orbit

  state.resize(num_camera_orbits + 4, 0.5f);
}

void MVFCN::train()
{
  // get list of mesh filenames in an input directory
  THEA_CONSOLE << "\n*** Entering training mode *** \n";
  if (!FileSystem::directoryExists(Settings::train_meshes_path))
  {
    THEA_ERROR << "Specified training directory " << Settings::train_meshes_path << " does not exist.";
    return;
  }
  vector<std::string> list_mesh_paths;
  if (FileSystem::getDirectoryContents(Settings::train_meshes_path, list_mesh_paths, FileSystem::ObjectType::FILE, "*.off *.obj  *.3ds") == 0)
  {
    THEA_ERROR << "Specified training directory " << Settings::train_meshes_path << " has no OFF, OBJ or 3DS meshes.";
    return;
  }
  if (!createAuxiliaryDirectories(Settings::train_meshes_path, Settings::skip_train_rendering))
  {
    THEA_ERROR << "Auxiliary directories cannot be created - cannot proceed with training.";
    return;
  }


  // read any pre-existing labels 
  label_map.clear();
  ifstream in_labels_file(FilePath::concat(Settings::train_meshes_path, LABELS_FILENAME));
  if (in_labels_file.good())
  {
    THEA_CONSOLE << "Found pre-existing labels file (perhaps from a previous rendering pass). Will load labels from that file.";
    while (!in_labels_file.eof())
      {
        string label;
        int id;
        if (in_labels_file.eof())
          break;
        in_labels_file >> label;
        if (in_labels_file.eof())
          break;
        in_labels_file >> id;
        label_map[label] = id;

        THEA_CONSOLE << "Read training label: " << label << " [id: " << id << "]";
      }
  }
  in_labels_file.close();


  // read any pre-existing validation data
  validation_map.clear();
  ifstream in_val_file(FilePath::concat(Settings::train_meshes_path, VALIDATION_DATA_FILENAME));
  if (in_val_file.good())
  {
    THEA_CONSOLE << "Found pre-existing validation file (perhaps from a previous rendering pass). Will load validation data from that file.";
    while (!in_val_file.eof())
    {
      string mesh_filename;
      bool used_for_validation;
      if (in_val_file.eof())
        break;
      in_val_file >> mesh_filename;
      if (in_val_file.eof())
        break;
      in_val_file >> used_for_validation;
      validation_map[mesh_filename] = used_for_validation;

      THEA_CONSOLE << "Mesh " << mesh_filename << " validation flag: " << used_for_validation;
    }
    in_val_file.close();
  }
  else // normally the val.txt would be written by the matlab script
  {
    vector<string> shuffled_list_mesh_paths = list_mesh_paths;
    std::random_shuffle(shuffled_list_mesh_paths.begin(), shuffled_list_mesh_paths.end());
    for (int m = 0; m < list_mesh_paths.size(); ++m)
    {
      if (m % 4 == 0) // every 4th mesh is used for validation (image filenames will have '@' appended in the end before the extension)
        validation_map[FilePath::baseName(shuffled_list_mesh_paths[m])] = true;
      else
        validation_map[FilePath::baseName(shuffled_list_mesh_paths[m])] = false;
    }
    in_val_file.close();
    ofstream out_val_file(FilePath::concat(Settings::train_meshes_path, VALIDATION_DATA_FILENAME));
    for (validation_map_iterator iter = validation_map.begin(); iter != validation_map.end(); iter++)
      out_val_file << iter->first << " " << iter->second << std::endl;
    out_val_file.close();
  }

  // process all input meshes
  meshes_processor_ptr.clear();
  THEA_CONSOLE << "Will process " << list_mesh_paths.size() << " meshes for training!";
  for (int m = 0; m < list_mesh_paths.size(); ++m)
  {
    std::shared_ptr<MeshProcessor> mesh_processor_ptr(new MeshProcessor(list_mesh_paths[m], true, label_map, !Settings::skip_train_rendering));
    if (validation_map[FilePath::baseName(list_mesh_paths[m])])
      mesh_processor_ptr->setValidationFlag();
    if (mesh_processor_ptr->getNumberOfFaces() > Settings::max_number_of_faces)
    {
      THEA_WARNING << "Mesh is way too big (#" << mesh_processor_ptr->getNumberOfFaces() << " faces) and will be ignored. You may overcome this warning by increasing the max-number-of-faces cmd line option (you may however run out of GPU mem)"; 
      mesh_processor_ptr.reset();
      continue;
    }
    if (!Settings::skip_train_rendering)
    {
      RenderViews rv(mesh_processor_ptr, Settings::train_meshes_path, label_map);
      rv.render();      
      if (!Settings::baseline_rendering)  // in the case if multi-scale rendering, adding additional dodecahedron views as one more camera orbit helps a bit
      {
        Settings::baseline_rendering = true;
        RenderViews rv(mesh_processor_ptr, Settings::train_meshes_path, label_map, Settings::num_cam_distances);
        rv.render();
        Settings::baseline_rendering = false;
      }
    }
    if (Settings::do_only_rendering)
    {
      mesh_processor_ptr->freeMeshData();
      mesh_processor_ptr.reset();
      THEA_CONSOLE << "Done rendering " << list_mesh_paths[m];
      continue;
    }
#ifndef SKIP_COMPILING_CAFFE_NETWORK_CODE
    mesh_processor_ptr->computeMeshPairwiseFeatures(FilePath::concat(FilePath::concat(Settings::train_meshes_path, OUTPUT_MESH_METADATA_FOLDER), FilePath::baseName(mesh_processor_ptr->getMeshPath()) + "_crf_pairwise_features.bin"), false);
#endif
    mesh_processor_ptr->freeMeshData();
    meshes_processor_ptr.push_back(mesh_processor_ptr);

    THEA_CONSOLE << "Done processing " << list_mesh_paths[m];
  }

  ofstream out_labels_file(FilePath::concat(Settings::train_meshes_path, LABELS_FILENAME));
  for (label_map_iterator iter = label_map.begin(); iter != label_map.end(); iter++)
    out_labels_file << iter->first << " " << iter->second << std::endl;
  out_labels_file.close();

  if (meshes_processor_ptr.empty())
    return;

#ifndef SKIP_COMPILING_CAFFE_NETWORK_CODE  
  crf = std::shared_ptr<MeshCRF>(new MeshCRF(label_map, FilePath::concat(Settings::train_meshes_path, OUTPUT_MESH_METADATA_FOLDER), meshes_processor_ptr[0]->getNumberOfPairwiseFeatures()));  
  loadState(FilePath::concat(FilePath::concat(Settings::train_meshes_path, LEARNING_METADATA_FOLDER), STATE_FILENAME));

  if (!Settings::skip_fcn)
  {
    if (!Settings::skip_fcn_train) // first training pass (find #iterations to use, validation viewpoint configuratiion)
    {
      if ((int)state[num_camera_orbits + 2] < 1)
      {
        fcntrain(label_map.size(), Settings::train_meshes_path, false);
        fcntest(label_map.size(), Settings::train_meshes_path, Settings::train_meshes_path, false);
        saveState(FilePath::concat(FilePath::concat(Settings::train_meshes_path, LEARNING_METADATA_FOLDER), STATE_FILENAME));
        deleteShapshots(FilePath::concat(Settings::train_meshes_path, LEARNING_METADATA_FOLDER), "frontend_vgg_model_iter_");
      }
      fcntrain(label_map.size(), Settings::train_meshes_path, true);  // second pass including validation data
    }

    fcntest(label_map.size(), Settings::train_meshes_path, Settings::train_meshes_path, false, true);      
    saveState(FilePath::concat(FilePath::concat(Settings::train_meshes_path, LEARNING_METADATA_FOLDER), STATE_FILENAME));
  }

  string crf_parameter_file = FilePath::concat(FilePath::concat(Settings::train_meshes_path, LEARNING_METADATA_FOLDER), DISJOINT_CRF_PARAMETERS_FILENAME);
  if (!Settings::skip_crf_train)
  {
    crf->train(meshes_processor_ptr, crf_parameter_file);
    crf->outputCRFParameters(FilePath::concat(FilePath::concat(Settings::train_meshes_path, LEARNING_METADATA_FOLDER), CRF_PARAMETERS_FILENAME)); // will be overwritten by mvfcntrain
  }
  if( !crf->loadCRFParameters(crf_parameter_file) )
    THEA_WARNING << "Could not import learned crf parameters from " << crf_parameter_file << ". Will use default non-optimized parameters during testing.";
  crf->isItCalledFromJointMVFCNModel(false);

  for (int m = 0; m < meshes_processor_ptr.size(); ++m)
  {
    THEA_CONSOLE << "Checking learned model on training mesh " << m + 1 << "/" << meshes_processor_ptr.size() << ": " << meshes_processor_ptr[m]->getMeshPath() << "...";    
    mesh_labeling_accuracies.push_back(crf->mfinference(meshes_processor_ptr[m]));
    meshes_processor_ptr[m]->freeMeshCRFData();
  }
  outputMeshLabelingAccuracies(Settings::train_meshes_path, false, true);
  saveState(FilePath::concat(FilePath::concat(Settings::train_meshes_path, LEARNING_METADATA_FOLDER), STATE_FILENAME));
  mesh_labeling_accuracies.clear();

  if (!Settings::skip_mvfcn)
  {    
    if (!Settings::skip_mvfcn_train)
      mvfcntrain(label_map.size(), Settings::train_meshes_path, !Settings::do_not_use_crf_mvfcn);
    fcntest(label_map.size(), Settings::train_meshes_path, Settings::train_meshes_path, true);
    saveState(FilePath::concat(FilePath::concat(Settings::train_meshes_path, LEARNING_METADATA_FOLDER), STATE_FILENAME));
  }

  crf_parameter_file = FilePath::concat(FilePath::concat(Settings::train_meshes_path, LEARNING_METADATA_FOLDER), CRF_PARAMETERS_FILENAME);
  if (!crf->loadCRFParameters(crf_parameter_file))
    THEA_WARNING << "Could not import learned crf parameters from " << crf_parameter_file << ". Will use default non-optimized parameters during testing.";
  crf->isItCalledFromJointMVFCNModel(true);

  for (int m = 0; m < meshes_processor_ptr.size(); ++m)
  {
    THEA_CONSOLE << "Checking jointly learned model on training mesh " << m + 1 << "/" << meshes_processor_ptr.size() << ": " << meshes_processor_ptr[m]->getMeshPath() << "...";    
    mesh_labeling_accuracies.push_back(crf->mfinference(meshes_processor_ptr[m]));
    meshes_processor_ptr[m]->freeMeshCRFData();
  }
  outputMeshLabelingAccuracies(Settings::train_meshes_path, true, true);
  saveState(FilePath::concat(FilePath::concat(Settings::train_meshes_path, LEARNING_METADATA_FOLDER), STATE_FILENAME));

#endif
}




void MVFCN::test()
{
  // get list of mesh filenames in an input directory
  THEA_CONSOLE << "\n*** Entering testing mode *** \n";
  if (!FileSystem::directoryExists(Settings::test_meshes_path))
  {
    THEA_ERROR << "Specified test directory " << Settings::test_meshes_path << " does not exist.";
    return;
  }
  vector<std::string> list_mesh_paths;
  if (FileSystem::getDirectoryContents(Settings::test_meshes_path, list_mesh_paths, FileSystem::ObjectType::FILE, "*.off *.obj  *.3ds") == 0)
  {
    THEA_ERROR << "Specified test directory " << Settings::test_meshes_path << " has no OFF, OBJ or 3DS meshes.";
    return;
  }
  if (!createAuxiliaryDirectories(Settings::test_meshes_path, Settings::skip_test_rendering))
  {
    THEA_ERROR << "Auxiliary directories cannot be created - cannot proceed with testing.";
    return;
  }

  // read labels 
  label_map.clear();
  ifstream in_labels_file(FilePath::concat(Settings::train_meshes_path, LABELS_FILENAME));
  if (!in_labels_file.good())
  {
    THEA_ERROR << "Cannot proceed: cannot read label file (storing all training label names & ids): " << FilePath::concat(Settings::train_meshes_path, LABELS_FILENAME);
    return;
  }
  while (!in_labels_file.eof())
  {
    string label;
    int id;
    if (in_labels_file.eof())
      break;
    in_labels_file >> label;
    if (in_labels_file.eof())
      break;
    in_labels_file >> id;
    label_map[label] = id;

    THEA_CONSOLE << "Read training label: " << label << " [id: " << id << "]";
  }
  in_labels_file.close();

  // process all input meshes, find all available part labels
  THEA_CONSOLE << "Will process " << list_mesh_paths.size() << " meshes for testing";
  meshes_processor_ptr.clear();
  mesh_labeling_accuracies.clear();
  for (int m = 0; m < list_mesh_paths.size(); ++m)
  {
    std::shared_ptr<MeshProcessor> mesh_processor_ptr(new MeshProcessor(list_mesh_paths[m], false, label_map, !Settings::skip_test_rendering));
    if (mesh_processor_ptr->getNumberOfFaces() > Settings::max_number_of_faces)
    {
      THEA_WARNING << "Mesh is way too big (#" << mesh_processor_ptr->getNumberOfFaces() << " faces) and will be ignored. You may overcome this warning by increasing the max-number-of-faces cmd line option (you may however run out of GPU mem)";
      mesh_processor_ptr.reset();
      continue;
    }
    if (Settings::train_meshes_path != Settings::test_meshes_path)
    {
      if (!Settings::skip_test_rendering)
      {
        RenderViews rv(mesh_processor_ptr, Settings::test_meshes_path, label_map);
        rv.render();
        if (!Settings::baseline_rendering)  // in the case if multi-scale rendering, adding additional dodecahedron views as one more camera orbit helps a bit
        {
          Settings::baseline_rendering = true;
          RenderViews rv(mesh_processor_ptr, Settings::test_meshes_path, label_map, Settings::num_cam_distances);
          rv.render();
          Settings::baseline_rendering = false;
        }

      }
    }
    if (Settings::do_only_rendering)
    {
      mesh_processor_ptr->freeMeshData();      
      mesh_processor_ptr.reset();
      THEA_CONSOLE << "Done rendering " << list_mesh_paths[m];
      continue;
    }
#ifndef SKIP_COMPILING_CAFFE_NETWORK_CODE      
    mesh_processor_ptr->computeMeshPairwiseFeatures(FilePath::concat(FilePath::concat(Settings::test_meshes_path, OUTPUT_MESH_METADATA_FOLDER), FilePath::baseName(mesh_processor_ptr->getMeshPath()) + "_crf_pairwise_features.bin"), false);
#endif
    mesh_processor_ptr->freeMeshData();
    meshes_processor_ptr.push_back(mesh_processor_ptr);

    THEA_CONSOLE << "Done processing " << list_mesh_paths[m];
  }
  if (meshes_processor_ptr.empty())
    return;

#ifndef SKIP_COMPILING_CAFFE_NETWORK_CODE   
  crf = std::shared_ptr<MeshCRF>(new MeshCRF(label_map, FilePath::concat(Settings::test_meshes_path, OUTPUT_MESH_METADATA_FOLDER), meshes_processor_ptr[0]->getNumberOfPairwiseFeatures()));
  loadState(FilePath::concat(FilePath::concat(Settings::train_meshes_path, LEARNING_METADATA_FOLDER), STATE_FILENAME));
 
  if (!Settings::skip_fcn)
    fcntest(label_map.size(), Settings::train_meshes_path, Settings::test_meshes_path, false); 

  string crf_parameter_file = FilePath::concat(FilePath::concat(Settings::train_meshes_path, LEARNING_METADATA_FOLDER), DISJOINT_CRF_PARAMETERS_FILENAME);
  if (!crf->loadCRFParameters(crf_parameter_file))
    THEA_WARNING << "Could not import learned crf parameters from " << crf_parameter_file << ". Will use default non-optimized parameters during testing.";
  crf->isItCalledFromJointMVFCNModel(false);

  for (int m = 0; m < meshes_processor_ptr.size(); ++m)
  {
    THEA_CONSOLE << "Testing mesh " << m + 1 << "/" << meshes_processor_ptr.size() << ": " << meshes_processor_ptr[m]->getMeshPath() << "...";
    mesh_labeling_accuracies.push_back(crf->mfinference(meshes_processor_ptr[m], 100));
    meshes_processor_ptr[m]->freeMeshCRFData();
  }
  outputMeshLabelingAccuracies(Settings::test_meshes_path, false, false);
  mesh_labeling_accuracies.clear();

  if (!Settings::skip_mvfcn) // will overwrite the above fcntest results
    fcntest(label_map.size(), Settings::train_meshes_path, Settings::test_meshes_path, state[num_camera_orbits + 1] <= state[num_camera_orbits]);

  if (state[num_camera_orbits + 1] <= state[num_camera_orbits])
    crf_parameter_file = FilePath::concat(FilePath::concat(Settings::train_meshes_path, LEARNING_METADATA_FOLDER), CRF_PARAMETERS_FILENAME);
  if (!crf->loadCRFParameters(crf_parameter_file))
    THEA_WARNING << "Could not import learned crf parameters from " << crf_parameter_file << ". Will use default non-optimized parameters during testing.";
  crf->isItCalledFromJointMVFCNModel(true);
  for (int m = 0; m < meshes_processor_ptr.size(); ++m)
  {
    THEA_CONSOLE << "Testing mesh " << m + 1 << "/" << meshes_processor_ptr.size() << ": " << meshes_processor_ptr[m]->getMeshPath() << "...";
    mesh_labeling_accuracies.push_back(crf->mfinference(meshes_processor_ptr[m], 100));
    meshes_processor_ptr[m]->freeMeshCRFData();
  }
  outputMeshLabelingAccuracies(Settings::test_meshes_path, true, false);
#endif
}





#ifndef SKIP_COMPILING_CAFFE_NETWORK_CODE
void MVFCN::fcntrain(const size_t num_classes, const string& train_dataset_path, bool include_validation_data_for_training)
{
  THEA_CONSOLE << "***** FCN PRE-TRAINING STARTS HERE *****";
  FLAGS_alsologtostderr = 1;
  get_gpus();
  std::shared_ptr<SolverParameter> solver_param(new SolverParameter());

  if (!google_logging_is_initialized)
  {
    ::google::InitGoogleLogging("mvfcn");
    google_logging_is_initialized = true;
  }


  // gather all training images, write image/list label list
  int num_total_training_images = 0;
  int num_total_validation_images = 0;
  for (int m = 0; m < meshes_processor_ptr.size(); ++m)
  {
    THEA_CONSOLE << "Searching for rendered images from mesh " << m + 1 << "/" << meshes_processor_ptr.size() << ": " << meshes_processor_ptr[m]->getMeshPath() << "...";
    vector<string> rendered_image_filenames = meshes_processor_ptr[m]->searchForImages(FilePath::concat(train_dataset_path, PRENDERED_IMAGES_FOLDER));
    if (rendered_image_filenames.empty()) // Check for invalid input
    {
      THEA_ERROR << "No rendered images found for mesh: " << meshes_processor_ptr[m]->getMeshPath();
      continue;
    }
    THEA_CONSOLE << "Found " << rendered_image_filenames.size() << " images.";
    num_total_training_images += rendered_image_filenames.size();
    if (meshes_processor_ptr[m]->isUsedForValidation())
      num_total_validation_images += rendered_image_filenames.size();
  }
  if (num_total_training_images == 0)
  {
    THEA_ERROR << "No rendered training images found! Cannot proceed with training!";
    return;
  }

  // patterns to find in the base model files
  boost::regex regex_last_layer_name("final");
  boost::regex regex_num_output_classes("num_output: 21");
  boost::regex regex_num_group_classes("group: 21");
  boost::regex regex_prendered_folder(PRENDERED_IMAGES_FOLDER);
  boost::regex regex_depth_folder(DEPTH_IMAGES_FOLDER);
  boost::regex regex_aux_folder(AUX_IMAGES_FOLDER); // SDF/UP change
  boost::regex regex_label_folder(SEGMENTATION_IMAGES_FOLDER);
  boost::regex regex_batch_size("batch_size: 4");
  boost::regex regex_validation_mode("validation_mode: 0");
  boost::regex regex_dim("dim: 900");

  boost::regex regex_train_net(BASE_TRAIN_MODEL_FILENAME);
  boost::regex regex_test_net(BASE_TEST_MODEL_FILENAME);
  boost::regex regex_model_net(OUTPUT_PRETRAINED_MODEL_FILENAME);
  boost::regex regex_solver_mode("solver_mode: GPU");
  boost::regex regex_iter_size("iter_size: 8");
  boost::regex regex_test_iter("test_iter: 640");
  boost::regex regex_max_iter("max_iter: 1000");


  // train network definition
  std::ifstream train_network_file(BASE_TRAIN_MODEL_FILENAME);
  if (!train_network_file.good())
  {
    THEA_ERROR << "Cannot find base train model proto file: " << BASE_TRAIN_MODEL_FILENAME << " - make sure it is in the working path";
    exit(-1);
  }
  std::string train_network_definition((std::istreambuf_iterator<char>(train_network_file)), std::istreambuf_iterator<char>());
  train_network_definition = boost::regex_replace(train_network_definition, regex_last_layer_name, "final_new");
  train_network_definition = boost::regex_replace(train_network_definition, regex_num_output_classes, "num_output: " + std::to_string(num_classes));
  train_network_definition = boost::regex_replace(train_network_definition, regex_num_group_classes, "group: " + std::to_string(num_classes));
  train_network_definition = boost::regex_replace(train_network_definition, regex_batch_size, "batch_size: " + std::to_string(Settings::pretraining_batch_size / Settings::pretraining_batch_splits));
  if (include_validation_data_for_training)
    train_network_definition = boost::regex_replace(train_network_definition, regex_validation_mode, "validation_mode: 1");

  string prendered_folder = FilePath::concat(train_dataset_path, PRENDERED_IMAGES_FOLDER);
  std::replace(prendered_folder.begin(), prendered_folder.end(), '\\', '/');
  train_network_definition = boost::regex_replace(train_network_definition, regex_prendered_folder, prendered_folder);

  string depth_folder = FilePath::concat(train_dataset_path, DEPTH_IMAGES_FOLDER);
  std::replace(depth_folder.begin(), depth_folder.end(), '\\', '/');
  train_network_definition = boost::regex_replace(train_network_definition, regex_depth_folder, depth_folder);

  string aux_folder = FilePath::concat(train_dataset_path, AUX_IMAGES_FOLDER); // SDF/UP CHANGE
  std::replace(aux_folder.begin(), aux_folder.end(), '\\', '/');  // SDF/UP CHANGE
  train_network_definition = boost::regex_replace(train_network_definition, regex_aux_folder, aux_folder); // SDF/UP CHANGE

  string label_folder = FilePath::concat(train_dataset_path, SEGMENTATION_IMAGES_FOLDER);
  std::replace(label_folder.begin(), label_folder.end(), '\\', '/');
  train_network_definition = boost::regex_replace(train_network_definition, regex_label_folder, label_folder);

  std::ofstream output_train_network_file(FilePath::concat(FilePath::concat(train_dataset_path, LEARNING_METADATA_FOLDER), BASE_TRAIN_MODEL_FILENAME));
  output_train_network_file << train_network_definition << std::endl;
  train_network_file.close();
  output_train_network_file.close();

  // test network definition
  std::ifstream test_network_file(BASE_TEST_MODEL_FILENAME);
  if (!test_network_file.good())
  {
    THEA_ERROR << "Cannot find base test model proto file: " << BASE_TEST_MODEL_FILENAME << " - make sure it is in the working path";
    exit(-1);
  }
  std::string test_network_definition((std::istreambuf_iterator<char>(test_network_file)), std::istreambuf_iterator<char>());
  test_network_definition = boost::regex_replace(test_network_definition, regex_last_layer_name, "final_new");
  test_network_definition = boost::regex_replace(test_network_definition, regex_num_output_classes, "num_output: " + std::to_string(num_classes));
  test_network_definition = boost::regex_replace(test_network_definition, regex_num_group_classes, "group: " + std::to_string(num_classes));
  test_network_definition = boost::regex_replace(test_network_definition, regex_prendered_folder, prendered_folder);
  test_network_definition = boost::regex_replace(test_network_definition, regex_depth_folder, depth_folder);
  test_network_definition = boost::regex_replace(test_network_definition, regex_aux_folder, aux_folder); // SDF/UP CHANGE
  test_network_definition = boost::regex_replace(test_network_definition, regex_label_folder, label_folder);

  std::ofstream output_test_network_file(FilePath::concat(FilePath::concat(train_dataset_path, LEARNING_METADATA_FOLDER), BASE_TEST_MODEL_FILENAME));
  output_test_network_file << test_network_definition << std::endl;
  test_network_file.close();
  output_test_network_file.close();

  // deploy network definition
  std::ifstream deploy_network_file(BASE_DEPLOY_MODEL_FILENAME);
  if (!deploy_network_file.good())
  {
    THEA_ERROR << "Cannot find base deploy model proto file: " << BASE_DEPLOY_MODEL_FILENAME << " - make sure it is in the working path";
    exit(-1);
  }
  std::string deploy_network_definition((std::istreambuf_iterator<char>(deploy_network_file)), std::istreambuf_iterator<char>());
  deploy_network_definition = boost::regex_replace(deploy_network_definition, regex_last_layer_name, "final_new");
  deploy_network_definition = boost::regex_replace(deploy_network_definition, regex_num_output_classes, "num_output: " + std::to_string(num_classes));
  deploy_network_definition = boost::regex_replace(deploy_network_definition, regex_num_group_classes, "group: " + std::to_string(num_classes));
  deploy_network_definition = boost::regex_replace(deploy_network_definition, regex_dim, "dim: " + std::to_string(Settings::render_size + 2 * label_margin));

  std::ofstream output_deploy_network_file(FilePath::concat(FilePath::concat(train_dataset_path, LEARNING_METADATA_FOLDER), BASE_DEPLOY_MODEL_FILENAME));
  output_deploy_network_file << deploy_network_definition << std::endl;
  deploy_network_file.close();
  output_deploy_network_file.close();

  // solver definition
  std::ifstream solver_file(BASE_SOLVER_FILENAME);
  if (!solver_file.good())
  {
    THEA_ERROR << "Cannot find base solver file: " << BASE_SOLVER_FILENAME << " - make sure it is in the working path";
    exit(-1);
  }
  std::string solver_definition((std::istreambuf_iterator<char>(solver_file)), std::istreambuf_iterator<char>());
  solver_file.close();

  string train_net_file = FilePath::concat(FilePath::concat(train_dataset_path, LEARNING_METADATA_FOLDER), BASE_TRAIN_MODEL_FILENAME);
  std::replace(train_net_file.begin(), train_net_file.end(), '\\', '/');
  solver_definition = boost::regex_replace(solver_definition, regex_train_net, train_net_file);

  string test_net_file = FilePath::concat(FilePath::concat(train_dataset_path, LEARNING_METADATA_FOLDER), BASE_TEST_MODEL_FILENAME);
  std::replace(test_net_file.begin(), test_net_file.end(), '\\', '/');
  solver_definition = boost::regex_replace(solver_definition, regex_test_net, test_net_file);

  string output_model_file = FilePath::concat(FilePath::concat(train_dataset_path, LEARNING_METADATA_FOLDER), OUTPUT_PRETRAINED_MODEL_FILENAME);
  std::replace(output_model_file.begin(), output_model_file.end(), '\\', '/');
  solver_definition = boost::regex_replace(solver_definition, regex_model_net, output_model_file);

  int pretraining_num_iterations = (int)( ( (float)num_total_training_images / (float)Settings::pretraining_batch_size ) * (float)Settings::pretraining_num_epochs );

  solver_definition = boost::regex_replace(solver_definition, regex_iter_size, "iter_size: " + std::to_string(Settings::pretraining_batch_splits));
  solver_definition = boost::regex_replace(solver_definition, regex_test_iter, "test_iter: " + std::to_string(num_total_validation_images)); // for evaluating validation error
  if (include_validation_data_for_training && (int)state[num_camera_orbits + 2] >= 1)
    solver_definition = boost::regex_replace(solver_definition, regex_max_iter, "max_iter: " + std::to_string((int)state[num_camera_orbits + 2])); // use num iterations from previous training round (set according to validation)
  else
    solver_definition = boost::regex_replace(solver_definition, regex_max_iter, "max_iter: " + std::to_string(pretraining_num_iterations)); // use predefined num iterations
  if (Settings::gpu_use == "false")
  {
    solver_definition = boost::regex_replace(solver_definition, regex_solver_mode, "solver_mode: CPU");
  }
  google::protobuf::TextFormat::ParseFromString(solver_definition, solver_param.get());

  if (gpus.empty() || Settings::gpu_use == "false")
  {
    THEA_CONSOLE << "Will use CPU [slow!]";
    Caffe::set_mode(Caffe::CPU);
  }
  else {
    ostringstream s;
    for (int i = 0; i < gpus.size(); ++i)
    {
      s << (i ? ", " : "") << gpus[i];
    }
    THEA_CONSOLE << "Will use GPUs: " << s.str();
    cudaDeviceProp device_prop;
    for (int i = 0; i < gpus.size(); ++i)
    {
      cudaGetDeviceProperties(&device_prop, gpus[i]);
      THEA_CONSOLE << "GPU " << gpus[i] << ": " << device_prop.name;
    }

    solver_param->set_device_id(gpus[0]);
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
    Caffe::set_solver_count((int)gpus.size());
  }

  boost::shared_ptr<caffe::Solver<float> > solver = boost::shared_ptr<caffe::Solver<float> >(caffe::SolverRegistry<float>::CreateSolver(*solver_param));
  // solver->Restore(snapshot.c_str());
  if (!Settings::do_not_use_pretrained_model)
  {
    THEA_CONSOLE << "Will use pre-trained VGG model...";
    NetParameter pretrained_param;
    ReadNetParamsFromBinaryFileOrDie(PRETRAINED_MODEL_FILENAME, &pretrained_param);
    solver->net()->CopyTrainedLayersFromAndResizeChannelsByAveragingIfNecessary(pretrained_param);
    //solver->net()->CopyTrainedLayersFrom(PRETRAINED_MODEL_FILENAME);
  }
  string snapshot_filename = findLatestShapshot(FilePath::concat(train_dataset_path, LEARNING_METADATA_FOLDER), "frontend_vgg_model_iter_");
  if (snapshot_filename != "")
    solver->Restore(snapshot_filename.c_str());

  THEA_CONSOLE << "Starting optimization...";
  if (gpus.size() > 1)
  {
    caffe::P2PSync<float> sync(solver, NULL, solver->param());
    sync.Run(gpus);
  }
  else
  {
    solver->Solve();
  }

  solver->net()->ToHDF5(output_model_file + "_iter_" + std::to_string(Settings::pretraining_num_epochs) + ".hdf5");
  state[num_camera_orbits + 2] = (float)solver->best_iter_;
  state[num_camera_orbits + 3] = (float)solver->best_test_score_;
}
#endif





#ifndef SKIP_COMPILING_CAFFE_NETWORK_CODE
void MVFCN::mvfcntrain(const size_t num_classes, const string& train_dataset_path, const bool use_crf_mvfcn)
{
  THEA_CONSOLE << "***** JOINT MVFCN TRAINING STARTS HERE *****";
  FLAGS_alsologtostderr = 1;
  get_gpus();
  std::shared_ptr<SolverParameter> solver_param(new SolverParameter());

  if (!google_logging_is_initialized)
  {
    ::google::InitGoogleLogging("mvfcn");
    google_logging_is_initialized = true;
  }

  // gather all training images, write image/list label list
  unsigned long max_num_pairwise_entries = 0;
  std::ofstream output_mesh_image_list_file(FilePath::concat(FilePath::concat(train_dataset_path, LEARNING_METADATA_FOLDER), MESH_LIST));
  for (int m = 0; m < meshes_processor_ptr.size(); ++m)
  {
    output_mesh_image_list_file << meshes_processor_ptr[m]->getMeshPath() << std::endl;
    max_num_pairwise_entries = std::max(max_num_pairwise_entries, meshes_processor_ptr[m]->getNumberOfEntriesInPairwiseFeatures());
  }
  output_mesh_image_list_file.close();


  // patterns to find in the base model files
  boost::regex regex_num_output_classes("num_output: 21");
  boost::regex regex_num_group_classes("group: 21");
  boost::regex regex_mesh_list_path(MESH_LIST);
  boost::regex regex_pretrained_parms_file(CRF_PARAMETERS_FILENAME);
  boost::regex regex_prendered_folder(PRENDERED_IMAGES_FOLDER);
  boost::regex regex_depth_folder(DEPTH_IMAGES_FOLDER);
  boost::regex regex_aux_folder(AUX_IMAGES_FOLDER); // SDF/UP CHANGE
  boost::regex regex_triangleid_folder(TRIANGLEID_IMAGES_FOLDER);
  boost::regex regex_crf_features_dir_folder(OUTPUT_MESH_METADATA_FOLDER);
  boost::regex regex_batch_size("batch_size: 4");
  boost::regex regex_stochastic("stochastic: false");
  boost::regex regex_max_num_faces("max_num_faces: 100000");
  boost::regex regex_max_num_pairwise_entries("max_num_pairwise_entries: 0");
  boost::regex regex_num_pairwise_features("num_pairwise_features: 0");

  boost::regex regex_train_net( string(MVFCN_TRAIN_MODEL_BASEFILENAME) + ".txt" );
  //boost::regex regex_test_net( string(MVFCN_TEST_MODEL_BASEFILENAME) + ".txt" );
  boost::regex regex_model_net(OUTPUT_MODEL_FILENAME);
  boost::regex regex_solver_mode("solver_mode: GPU");
  boost::regex regex_iter_size("iter_size: 8");
  boost::regex regex_max_iter("max_iter: 100");  


  // train network definition
  string train_network_filename;
  if (!Settings::do_not_use_stochastic_mvfcn)
    train_network_filename = string(MVFCN_TRAIN_MODEL_BASEFILENAME) + "_24";
  else if (Settings::baseline_rendering)
    train_network_filename = string(MVFCN_TRAIN_MODEL_BASEFILENAME) + "_80"; // not used anymore
  else
    train_network_filename = string(MVFCN_TRAIN_MODEL_BASEFILENAME) + "_" + std::to_string(Settings::max_images_per_distance * num_camera_orbits); // not used anymore

  if (use_crf_mvfcn)
    train_network_filename = train_network_filename + "_crf.txt";
  else
    train_network_filename = train_network_filename + "_nocrf.txt";

  std::ifstream train_network_file(train_network_filename);
  if (!train_network_file.good())
  {
    THEA_ERROR << "Cannot find base train model proto file: " << train_network_filename << " - make sure it is in the working path (or use createMVMeshFCNProto.m to generate it)";
    exit(-1);
  }
  std::string train_network_definition((std::istreambuf_iterator<char>(train_network_file)), std::istreambuf_iterator<char>());
  train_network_definition = boost::regex_replace(train_network_definition, regex_num_output_classes, "num_output: " + std::to_string(num_classes));
  train_network_definition = boost::regex_replace(train_network_definition, regex_num_group_classes, "group: " + std::to_string(num_classes));
  train_network_definition = boost::regex_replace(train_network_definition, regex_max_num_faces, "max_num_faces: " + std::to_string( Settings::max_number_of_faces ));
  train_network_definition = boost::regex_replace(train_network_definition, regex_max_num_pairwise_entries, "max_num_pairwise_entries: " + std::to_string(max_num_pairwise_entries));
  if (use_crf_mvfcn)
    train_network_definition = boost::regex_replace(train_network_definition, regex_num_pairwise_features, "num_pairwise_features: " + std::to_string(meshes_processor_ptr[0]->getNumberOfPairwiseFeatures()));
  else
    train_network_definition = boost::regex_replace(train_network_definition, regex_num_pairwise_features, "num_pairwise_features: 0");
  train_network_definition = boost::regex_replace(train_network_definition, regex_batch_size, "batch_size: " + std::to_string(Settings::training_batch_size / Settings::training_batch_splits));

  string train_mesh_list_filename = FilePath::concat(FilePath::concat(train_dataset_path, LEARNING_METADATA_FOLDER), MESH_LIST);
  std::replace(train_mesh_list_filename.begin(), train_mesh_list_filename.end(), '\\', '/');
  train_network_definition = boost::regex_replace(train_network_definition, regex_mesh_list_path, train_mesh_list_filename);

  string crf_pretrained_parms_file = FilePath::concat(FilePath::concat(train_dataset_path, LEARNING_METADATA_FOLDER), CRF_PARAMETERS_FILENAME);
  std::replace(crf_pretrained_parms_file.begin(), crf_pretrained_parms_file.end(), '\\', '/');
  train_network_definition = boost::regex_replace(train_network_definition, regex_pretrained_parms_file, crf_pretrained_parms_file);

  string prendered_folder = FilePath::concat(train_dataset_path, PRENDERED_IMAGES_FOLDER);
  std::replace(prendered_folder.begin(), prendered_folder.end(), '\\', '/');
  train_network_definition = boost::regex_replace(train_network_definition, regex_prendered_folder, prendered_folder);

  string depth_folder = FilePath::concat(train_dataset_path, DEPTH_IMAGES_FOLDER);
  std::replace(depth_folder.begin(), depth_folder.end(), '\\', '/');
  train_network_definition = boost::regex_replace(train_network_definition, regex_depth_folder, depth_folder);

  string aux_folder = FilePath::concat(train_dataset_path, AUX_IMAGES_FOLDER); // SDF/UP change
  std::replace(aux_folder.begin(), aux_folder.end(), '\\', '/'); // SDF/UP change
  train_network_definition = boost::regex_replace(train_network_definition, regex_aux_folder, aux_folder); // SDF/UP change

  string triangleid_folder = FilePath::concat(train_dataset_path, TRIANGLEID_IMAGES_FOLDER);
  std::replace(triangleid_folder.begin(), triangleid_folder.end(), '\\', '/');
  train_network_definition = boost::regex_replace(train_network_definition, regex_triangleid_folder, triangleid_folder);

  string crf_features_dir_folder = FilePath::concat(train_dataset_path, OUTPUT_MESH_METADATA_FOLDER);
  std::replace(crf_features_dir_folder.begin(), crf_features_dir_folder.end(), '\\', '/');
  train_network_definition = boost::regex_replace(train_network_definition, regex_crf_features_dir_folder, crf_features_dir_folder);

  if (!Settings::do_not_use_stochastic_mvfcn)
    train_network_definition = boost::regex_replace(train_network_definition, regex_stochastic, "stochastic: true");

  std::ofstream output_train_network_file(FilePath::concat(FilePath::concat(train_dataset_path, LEARNING_METADATA_FOLDER), train_network_filename ));
  output_train_network_file << train_network_definition << std::endl;
  train_network_file.close();
  output_train_network_file.close();

  // solver definition
  std::ifstream solver_file(MVFCN_SOLVER_FILENAME);
  if (!solver_file.good())
  {
    THEA_ERROR << "Cannot find base solver file: " << MVFCN_SOLVER_FILENAME << " - make sure it is in the working path";
    exit(-1);
  }
  std::string solver_definition((std::istreambuf_iterator<char>(solver_file)), std::istreambuf_iterator<char>());
  solver_file.close();

  string train_net_file = FilePath::concat(FilePath::concat(train_dataset_path, LEARNING_METADATA_FOLDER), train_network_filename);
  std::replace(train_net_file.begin(), train_net_file.end(), '\\', '/');
  solver_definition = boost::regex_replace(solver_definition, regex_train_net, train_net_file);

  string output_model_file = FilePath::concat(FilePath::concat(train_dataset_path, LEARNING_METADATA_FOLDER), OUTPUT_MODEL_FILENAME);
  std::replace(output_model_file.begin(), output_model_file.end(), '\\', '/');
  solver_definition = boost::regex_replace(solver_definition, regex_model_net, output_model_file);

  int training_num_iterations = (int)(((float)meshes_processor_ptr.size() / (float)Settings::training_batch_size) * (float)(Settings::training_num_epochs));

  solver_definition = boost::regex_replace(solver_definition, regex_iter_size, "iter_size: " + std::to_string(Settings::training_batch_splits));
  solver_definition = boost::regex_replace(solver_definition, regex_max_iter, "max_iter: " + std::to_string(training_num_iterations));

  if (gpus.empty() || Settings::gpu_use == "false" || Settings::do_not_use_stochastic_mvfcn)
  {
    solver_definition = boost::regex_replace(solver_definition, regex_solver_mode, "solver_mode: CPU");
  }
  google::protobuf::TextFormat::ParseFromString(solver_definition, solver_param.get());

  if (gpus.empty() || Settings::gpu_use == "false" || Settings::do_not_use_stochastic_mvfcn)
  {
    THEA_CONSOLE << "Will use CPU [slow!]";
    Caffe::set_mode(Caffe::CPU);
  }
  else {
    ostringstream s;
    for (int i = 0; i < gpus.size(); ++i)
    {
      s << (i ? ", " : "") << gpus[i];
    }
    THEA_CONSOLE << "Will use GPUs: " << s.str();
    cudaDeviceProp device_prop;
    for (int i = 0; i < gpus.size(); ++i)
    {
      cudaGetDeviceProperties(&device_prop, gpus[i]);
      THEA_CONSOLE << "GPU " << gpus[i] << ": " << device_prop.name;
    }

    solver_param->set_device_id(gpus[0]);
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
    Caffe::set_solver_count((int)gpus.size());
  }

  boost::shared_ptr<caffe::Solver<float> > solver = boost::shared_ptr<caffe::Solver<float> >(caffe::SolverRegistry<float>::CreateSolver(*solver_param));
  solver->net()->CopyTrainedLayersFromHDF5(FilePath::concat(FilePath::concat(train_dataset_path, LEARNING_METADATA_FOLDER), string(OUTPUT_PRETRAINED_MODEL_FILENAME) + "_iter_" + std::to_string(Settings::pretraining_num_epochs) + ".hdf5"));

  string snapshot_filename = findLatestShapshot(FilePath::concat(train_dataset_path, LEARNING_METADATA_FOLDER), "mvfcn_model_iter_");
  if (snapshot_filename != "")
    solver->Restore(snapshot_filename.c_str());

  THEA_CONSOLE << "Starting optimization...";
  if (gpus.size() > 1)
  {
    caffe::P2PSync<float> sync(solver, NULL, solver->param());
    sync.Run(gpus);
  }
  else
  {
    solver->Solve();
  }

  solver->net()->ToHDF5(output_model_file + "_iter_" + std::to_string(Settings::training_num_epochs) + ".hdf5");
}
#endif



#ifndef SKIP_COMPILING_CAFFE_NETWORK_CODE
bool MVFCN::fcntest(const size_t num_classes, const string& train_dataset_path, const string& test_dataset_path, const bool use_mvfcn_model, const bool do_not_check_views)
{
  if (train_dataset_path != test_dataset_path)
    THEA_CONSOLE << "***** TESTING STARTS HERE *****";
  else
    THEA_CONSOLE << "***** EVALUATING MODEL ON TRAINING DATA *****";
  FLAGS_alsologtostderr = 1;
  get_gpus();
  if (gpus.empty() || Settings::gpu_use == "false")
  {
    THEA_CONSOLE << "Will use CPU [slow!]";
    Caffe::set_mode(Caffe::CPU);
  }
  else
  {
    THEA_CONSOLE << "Will use GPU: " << gpus[0];
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, gpus[0]);
    THEA_CONSOLE << "GPU " << gpus[0] << ": " << device_prop.name;
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  }

  if (!google_logging_is_initialized)
  {
    ::google::InitGoogleLogging("mvfcn");
    google_logging_is_initialized = true;
  }

  // Instantiate the caffe net.
  std::shared_ptr<Net<float> > net;
  net.reset(new Net<float>(FilePath::concat(FilePath::concat(train_dataset_path, LEARNING_METADATA_FOLDER), BASE_DEPLOY_MODEL_FILENAME), TEST));
  if (use_mvfcn_model)
    net->CopyTrainedLayersFromHDF5(FilePath::concat(FilePath::concat(train_dataset_path, LEARNING_METADATA_FOLDER), string(OUTPUT_MODEL_FILENAME) + "_iter_" + std::to_string(Settings::training_num_epochs) + ".hdf5"));
  else
    net->CopyTrainedLayersFromHDF5(FilePath::concat(FilePath::concat(train_dataset_path, LEARNING_METADATA_FOLDER), string(OUTPUT_PRETRAINED_MODEL_FILENAME) + "_iter_" + std::to_string(Settings::pretraining_num_epochs) + ".hdf5"));

  THEA_CONSOLE << "Computing image-based label probabilities (CRF unary term) for " << meshes_processor_ptr.size() << " meshes.";
  float mean_image_accuracy = 0.0f;  
  vector<float> mean_image_accuracy_per_view(num_camera_orbits, 0.0f);
  vector<float> num_images_with_accuracy_per_view(num_camera_orbits, 0.0f);
  vector< vector <string> > rendered_image_filenames_per_view(num_camera_orbits);

  for (int m = 0; m < meshes_processor_ptr.size(); ++m)
  {
    THEA_CONSOLE << "Testing on images of mesh " << m + 1 << "/" << meshes_processor_ptr.size() << ": " << meshes_processor_ptr[m]->getMeshPath() << "...";
    float num_mesh_active_images = 0.0f;
    float mean_mesh_image_accuracy = 0.0f;
    vector<string> rendered_image_filenames = meshes_processor_ptr[m]->searchForImages(FilePath::concat(test_dataset_path, PRENDERED_IMAGES_FOLDER));
    if (rendered_image_filenames.empty()) // Check for invalid input
    {
      THEA_ERROR << "No rendered images found for mesh: " << meshes_processor_ptr[m]->getMeshPath();
      continue;
    }
    meshes_processor_ptr[m]->initFaceLabelProbabilities(num_classes, view_pooling_type);

    for (int i = 0; i < rendered_image_filenames.size(); i++)
    {
      // recognize view id of the file
      int view_id = -1;
      std::size_t found_ = rendered_image_filenames[i].find_last_of("_");
      if (found_ == string::npos || found_ == 0)
        THEA_WARNING << "Camera id could not be recognized by filename";
      else
      {
        string cropped_rendered_image_filename = rendered_image_filenames[i].substr(0, found_);
        std::size_t found_ = cropped_rendered_image_filename.find_last_of("_");
        if (found_ == string::npos)
          THEA_WARNING << "Camera id could not be recognized by filename";
        else
          view_id = stoi(cropped_rendered_image_filename.substr(found_ + 1));
      }
      if (view_id >= 0 && view_id < num_camera_orbits)
      {
        if (state[view_id] < 0.0f)
        {
          continue;
        }
        rendered_image_filenames_per_view[view_id].push_back(rendered_image_filenames[i]);
      }


      cv::Mat img = cv::imread(rendered_image_filenames[i], CV_LOAD_IMAGE_GRAYSCALE);
      if (!img.data) // Check for invalid input
      {
        THEA_ERROR << "Could not open or find the mesh rendered image: " << rendered_image_filenames[i];
        continue;
      }
      boost::regex regex_folder(PRENDERED_IMAGES_FOLDER);
      boost::regex regex_rendered_image_pattern(Thea::FilePath::baseName(meshes_processor_ptr[m]->getMeshPath()) + "_int_");
      string depth_image_filename = rendered_image_filenames[i];
      depth_image_filename = boost::regex_replace(depth_image_filename, regex_folder, DEPTH_IMAGES_FOLDER);
      depth_image_filename = boost::regex_replace(depth_image_filename, regex_rendered_image_pattern, Thea::FilePath::baseName(meshes_processor_ptr[m]->getMeshPath()) + "_dep_");
      cv::Mat depth_img = cv::imread(depth_image_filename, CV_LOAD_IMAGE_GRAYSCALE);
      string aux_image_filename = rendered_image_filenames[i]; // SDF/UP change
      aux_image_filename = boost::regex_replace(aux_image_filename, regex_folder, AUX_IMAGES_FOLDER);  // SDF/UP change
      aux_image_filename = boost::regex_replace(aux_image_filename, regex_rendered_image_pattern, Thea::FilePath::baseName(meshes_processor_ptr[m]->getMeshPath()) + "_aux_");  // SDF/UP change
      cv::Mat aux_img = cv::imread(aux_image_filename, CV_LOAD_IMAGE_GRAYSCALE);  // SDF/UP change

      Blob<float>* input_layer = net->input_blobs()[0];
      std::vector<cv::Mat> input_channels;
      float* input_data = input_layer->mutable_cpu_data();
      for (int c = 0; c < input_layer->channels(); ++c)
      {
        cv::Mat channel(input_layer->height(), input_layer->width(), CV_32F, input_data);
        input_channels.push_back(channel);
        input_data += input_layer->height() * input_layer->width();
      }

      /* Convert the input image to the input image format of the network. */
      cv::Mat sample_float;
      img.convertTo(sample_float, CV_32F);
      cv::Mat sample_normalized;
      cv::subtract(sample_float, image_mean, sample_normalized);
      cv::Mat sample_final;
      cv::copyMakeBorder(sample_normalized, sample_final, label_margin, label_margin, label_margin, label_margin, cv::BORDER_REFLECT101);
      //input_channels[0] = sample_final.clone();

      cv::Mat sample_float2;
      depth_img.convertTo(sample_float2, CV_32F);
      cv::Mat sample_normalized2;
      cv::subtract(sample_float2, depth_mean, sample_normalized2);
      cv::Mat sample_final2;
      cv::copyMakeBorder(sample_normalized2, sample_final2, label_margin, label_margin, label_margin, label_margin, cv::BORDER_REFLECT101);
      //input_channels[1] = sample_final2.clone();

      cv::Mat sample_float3; // SDF/UP change
      aux_img.convertTo(sample_float3, CV_32F);  // SDF/UP change
      cv::Mat sample_normalized3; // SDF/UP change
      cv::subtract(sample_float3, aux_mean, sample_normalized3); // SDF/UP change
      cv::Mat sample_final3; // SDF/UP change
      cv::copyMakeBorder(sample_normalized3, sample_final3, label_margin, label_margin, label_margin, label_margin, cv::BORDER_REFLECT101); // SDF/UP change
      //input_channels[2] = sample_final3.clone();


      /* This operation will write the separate planes directly to the
      * input layer of the network because it is wrapped by the cv::Mat
      * objects in input_channels. */
      //cv::split(sample_final, input_channels);
      cv::Mat input_image(input_layer->height(), input_layer->width(), CV_32FC3);
      vector<cv::Mat> input_image_channels(3); // FCN change
      input_image_channels[0] = sample_final;
      input_image_channels[1] = sample_final2;      // note: if you want to ignore depth, replace with sample_final here
      input_image_channels[2] = sample_final3;      // note: if you want to ignore aux, replace with sample_final here
      cv::merge(input_image_channels, input_image);
      cv::split(input_image, input_channels); // strange but it seems that it works only this way
      CHECK(reinterpret_cast<float*>(input_channels[0].data) == net->input_blobs()[0]->cpu_data()) << "Input channels are not wrapping the input layer of the network.";

      // inference
      net->Forward();

      ///* Copy the output layer to an image */
      Blob<float>* output_layer_blob = net->output_blobs()[0];
      std::vector<cv::Mat> output_channels;
      float* output_data = output_layer_blob->mutable_cpu_data();
      for (int c = 0; c < output_layer_blob->channels(); ++c)
      {
        cv::Mat channel(output_layer_blob->height(), output_layer_blob->width(), CV_32F, output_data);
        //cv::resize(channel, channel, cv::Size2i(Settings::render_size, Settings::render_size), CV_INTER_CUBIC); // if no deconvolution is used
        output_channels.push_back(channel);
        output_data += output_layer_blob->width() * output_layer_blob->height();
      }

      // attempt to find triangle id image
      string triangleID_image_filename = rendered_image_filenames[i];
      triangleID_image_filename = boost::regex_replace(triangleID_image_filename, regex_folder, TRIANGLEID_IMAGES_FOLDER);
      triangleID_image_filename = boost::regex_replace(triangleID_image_filename, regex_rendered_image_pattern, Thea::FilePath::baseName(meshes_processor_ptr[m]->getMeshPath()) + "_fid_");
      cv::Mat triangleID_img = cv::imread(triangleID_image_filename, CV_LOAD_IMAGE_UNCHANGED);
      if (!triangleID_img.data) // Check for invalid input
      {
        THEA_ERROR << "Could not open or find the triangle ID image, cannot project image to mesh: " << triangleID_image_filename;
        continue;
      }
      meshes_processor_ptr[m]->projectImageLabelProbabilitiesToMesh(output_channels, triangleID_img, view_pooling_type);

      // attempt to find ground-truth segmentation image (for measuring image-based accuracy)
      string ground_truth_segmentation_image_filename = rendered_image_filenames[i];
      ground_truth_segmentation_image_filename = boost::regex_replace(ground_truth_segmentation_image_filename, regex_folder, SEGMENTATION_IMAGES_FOLDER);
      ground_truth_segmentation_image_filename = boost::regex_replace(ground_truth_segmentation_image_filename, regex_rendered_image_pattern, Thea::FilePath::baseName(meshes_processor_ptr[m]->getMeshPath()) + "_lbl_");
      cv::Mat label_img = cv::imread(ground_truth_segmentation_image_filename, CV_LOAD_IMAGE_GRAYSCALE);
      if (!label_img.data) // Check for invalid input
      {
        THEA_WARNING << "Could not open or find the mesh ground truth segmentation image: " << ground_truth_segmentation_image_filename;
        continue;
      }

      // produce labeled images (for debugging)
      cv::Mat output_max_prob_image(Settings::render_size, Settings::render_size, CV_8U);
      for (int j = 0; j < Settings::render_size; ++j)
      {
        for (int k = 0; k < Settings::render_size; ++k)
        {
          if (label_img.at<unsigned char>(j, k) == 255) // background
          {
            output_max_prob_image.at<unsigned char>(j, k) = 255;
            continue;
          }
          float max_prob = 0.0f;
          for (int c = 0; c < output_layer_blob->channels(); ++c)
          {
            if (output_channels[c].at<float>(j, k) > max_prob)
            {
              max_prob = output_channels[c].at<float>(j, k);
              output_max_prob_image.at<unsigned char>(j, k) = (unsigned char)c;
            }
          }
        }
      }


      ///////////////////////////////////// DBG /////////////////////////////////////
      ///////////////////////////////////// DBG /////////////////////////////////////
      ///////////////////////////////////// DBG /////////////////////////////////////
      ///////////////////////////////////// DBG /////////////////////////////////////
      ///////////////////////////////////// DBG /////////////////////////////////////
//      if (view_id == 0 && i < 20)
//      {
//        vector<string> out_blob_names;
//        out_blob_names.push_back("conv1_1");
//        out_blob_names.push_back("conv2_1");
//        out_blob_names.push_back("conv3_1");
//        out_blob_names.push_back("conv4_1");
//        out_blob_names.push_back("conv5_1");
//        out_blob_names.push_back("fc6");
//        out_blob_names.push_back("fc7");
//
//        for (int bb = 0; bb < out_blob_names.size(); bb++)
//        {
//          boost::shared_ptr<Blob<float> > layer_blob = net->blob_by_name(out_blob_names[bb]);
//          std::vector<cv::Mat> output_channels;
//          float* layer_data = layer_blob->mutable_cpu_data();
//          for (int c = 0; c < min(layer_blob->channels(), 10); ++c)
//          {
//            cv::Mat channel(layer_blob->height(), layer_blob->width(), CV_32FC1, layer_data);
//            output_channels.push_back(channel);
//            layer_data += layer_blob->width() * layer_blob->height();
//
//            string channel_conv_id = "_" + out_blob_names[bb] + "_" + std::to_string(c);
//
////            cv::Mat tmp_img(layer_blob->height(), layer_blob->width(), CV_8UC4, output_channels[c].data);
//
//            cv::imwrite(FilePath::concat(FilePath::concat(test_dataset_path, OUTPUT_SEGMENTANTIONS_FOLDER), FilePath::baseName(rendered_image_filenames[i]) + channel_conv_id + ".exr"), output_channels[c]);
//          }
//        }
//      }
//
      ///////////////////////////////////// DBG /////////////////////////////////////
      ///////////////////////////////////// DBG /////////////////////////////////////
      ///////////////////////////////////// DBG /////////////////////////////////////
      ///////////////////////////////////// DBG /////////////////////////////////////
      ///////////////////////////////////// DBG /////////////////////////////////////


      // meassure image-based labeling accuracy
      float iter_accuracy = 0.0f;
      float count_non_background_pixels = 0.0f;
      for (int j = 0; j < Settings::render_size; ++j)
      {
        for (int k = 0; k < Settings::render_size; ++k)
        {
          if (label_img.at<unsigned char>(j, k) == 255)
          {
            continue;
          }
          count_non_background_pixels++;
          if (output_max_prob_image.at<unsigned char>(j, k) == label_img.at<unsigned char>(j, k))
          {
            iter_accuracy++;
          }
        }
      }

      cv::imwrite(FilePath::concat(FilePath::concat(test_dataset_path, OUTPUT_SEGMENTANTIONS_FOLDER), FilePath::baseName(rendered_image_filenames[i]) + ".png"), output_max_prob_image);
      iter_accuracy /= count_non_background_pixels;
      mean_mesh_image_accuracy += iter_accuracy;
      num_mesh_active_images++;
      if (view_id >= 0 && view_id < num_camera_orbits)
      {
        if (meshes_processor_ptr[m]->isUsedForValidation())
        {
          mean_image_accuracy_per_view[view_id] += iter_accuracy;
          num_images_with_accuracy_per_view[view_id]++;
        }
      }
       
      THEA_CONSOLE << "  + Tested image " << i + 1 << "/" << rendered_image_filenames.size() << ": " << FilePath::baseName(rendered_image_filenames[i]) << " [accuracy = " << 100.0f * iter_accuracy << "] (view id: " << view_id << ", ground truth seg. image: " << FilePath::baseName(ground_truth_segmentation_image_filename) << ")";
    } // end of mesh m processing


    mean_mesh_image_accuracy /= num_mesh_active_images;
    THEA_CONSOLE << "=> Mean image accuracy for mesh " << m + 1 << "/" << meshes_processor_ptr.size() << ": " << meshes_processor_ptr[m]->getMeshPath() << " [accuracy = " << 100.0f * mean_mesh_image_accuracy << "]";
    mean_image_accuracy += mean_mesh_image_accuracy;

    // unary term computation, saves all the features
    meshes_processor_ptr[m]->computeMeshNormalizedUnaryFeatures(FilePath::concat(FilePath::concat(test_dataset_path, OUTPUT_MESH_METADATA_FOLDER), FilePath::baseName(meshes_processor_ptr[m]->getMeshPath()) + "_crf_unary_features.txt"), true);
    meshes_processor_ptr[m]->freeMeshCRFData();
  }


  bool problematic_views_exist = false;
  float max_image_accuracy_across_views = 0.0f;
  for (int view_id = 0; view_id < num_camera_orbits; view_id++)
  {
    if (num_images_with_accuracy_per_view[view_id] > 0.5f)
    {
      mean_image_accuracy_per_view[view_id] /= (num_images_with_accuracy_per_view[view_id] + 1e-30f);
      THEA_CONSOLE << "Mean image accuracy in validation set for camera orbit  " << view_id << ": " << 100.0f * mean_image_accuracy_per_view[view_id];
      max_image_accuracy_across_views = std::max(max_image_accuracy_across_views, mean_image_accuracy_per_view[view_id]);
    }
  }

  if (train_dataset_path == test_dataset_path && !use_mvfcn_model && !do_not_check_views) // use only during training
  {
    for (int view_id = 0; view_id < num_camera_orbits; view_id++)
    {
      if (num_images_with_accuracy_per_view[view_id] > 0.5f)
      {
        if (max_image_accuracy_across_views - mean_image_accuracy_per_view[view_id] > 0.03f) // 3%
        {
          state[view_id] = -1.0f;
          THEA_CONSOLE << "Camera orbit " << view_id << " seems problematic according to validation set! Will remove all its images!!!";
          for (int f = 0; f < rendered_image_filenames_per_view[view_id].size(); f++)
          {
            THEA_CONSOLE << "Removing file: " << rendered_image_filenames_per_view[view_id][f] << std::endl;
            std::remove(rendered_image_filenames_per_view[view_id][f].c_str());
          }
          problematic_views_exist = true;
        }
      }
    }
  }

  mean_image_accuracy /= (float)meshes_processor_ptr.size();
  THEA_CONSOLE << std::endl << "Mean accuracy for all images: " << 100.0f * mean_image_accuracy;
  return !problematic_views_exist;
}


void MVFCN::outputMeshLabelingAccuracies(const string& dataset_path, const bool used_mvfcn_model, const bool called_from_training)
{
  if (meshes_processor_ptr.size() != mesh_labeling_accuracies.size())
  {
    THEA_ERROR << "Labeling accuracies were not computed for all meshes due to an internal error";
    return;
  }

  float mean_mesh_accuracy = 0.0f;
  string output_accuracy_filename = FilePath::concat(FilePath::concat(dataset_path, OUTPUT_MESH_METADATA_FOLDER), ACCURACY_FILENAME);
  if (used_mvfcn_model)
    output_accuracy_filename += "_mvfcn.txt";
  else
    output_accuracy_filename += "_fcn.txt";
  ofstream output_accuracy_file(output_accuracy_filename);
  THEA_CONSOLE << "The labeling accuracy for each mesh is the following";

  for (int m = 0; m < mesh_labeling_accuracies.size(); ++m)
  {
    output_accuracy_file << mesh_labeling_accuracies[m] << std::endl;
    mean_mesh_accuracy += mesh_labeling_accuracies[m];

    THEA_CONSOLE << meshes_processor_ptr[m]->getMeshPath() << ": " << mesh_labeling_accuracies[m];
  }

  mean_mesh_accuracy /= (float)mesh_labeling_accuracies.size();
  THEA_CONSOLE << "Mean mesh accuracy: " << 100.0f * mean_mesh_accuracy;
  output_accuracy_file << mean_mesh_accuracy << std::endl;
  output_accuracy_file.close();

  if (called_from_training)
  {
    if (used_mvfcn_model)
      state[num_camera_orbits + 1] = 1.0f - mean_mesh_accuracy;
    else
      state[num_camera_orbits] = (1.0f - mean_mesh_accuracy) + .005f;
  }
}


string MVFCN::findLatestShapshot(const string& search_path, const string& snapshot_base_name)
{
  const boost::regex pattern(snapshot_base_name + ".*" + ".state.h5");

  vector<string> matching_snapshot_filenames;

  boost::filesystem::directory_iterator end_itr; // Default ctor yields past-the-end
  for (boost::filesystem::directory_iterator i(search_path); i != end_itr; ++i)
  {
    // Skip if not a file
    if (!boost::filesystem::is_regular_file(i->status())) continue;

    // Skip if no match for V3:
    if (!boost::regex_match(i->path().filename().string(), pattern)) continue;

    // File matches, store it
    matching_snapshot_filenames.push_back(Thea::FilePath::concat(search_path, i->path().filename().string()));
  }

  string latest_snapshot_filename = "";
  int max_iter = -1;
  for (int f = 0; f < matching_snapshot_filenames.size(); ++f)
  {
    string::size_type pos = matching_snapshot_filenames[f].rfind(snapshot_base_name);
    if (pos == string::npos)
      continue;

    string matching_snapshot_filename_only_iter_and_extension = matching_snapshot_filenames[f].substr(pos + snapshot_base_name.length() );
    
    string::size_type pos2 = matching_snapshot_filename_only_iter_and_extension.find_last_of(".solverstate");
    if (pos2 == string::npos)
      continue;

    string str_iter = matching_snapshot_filename_only_iter_and_extension.substr(0, pos2);
    int iter = stoi(str_iter);
    if (iter > max_iter)
    {
      latest_snapshot_filename = matching_snapshot_filenames[f];
      max_iter = iter;
    }

    THEA_CONSOLE << "Found previous snapshot: " << matching_snapshot_filenames[f] << "(iter = " << iter << ")";

  }
  if (max_iter > -1)
    THEA_CONSOLE << "Latest snapshot: " << latest_snapshot_filename;
  else
    THEA_CONSOLE << "No previous snapshot found. Learning will start from scratch.";

  return latest_snapshot_filename;
}


void MVFCN::deleteShapshots(const string& search_path, const string& snapshot_base_name)
{
  const boost::regex pattern(snapshot_base_name + ".*");

  vector<string> matching_snapshot_filenames;

  boost::filesystem::directory_iterator end_itr; // Default ctor yields past-the-end
  for (boost::filesystem::directory_iterator i(search_path); i != end_itr; ++i)
  {
    // Skip if not a file
    if (!boost::filesystem::is_regular_file(i->status())) continue;

    // Skip if no match for V3:
    if (!boost::regex_match(i->path().filename().string(), pattern)) continue;

    // File matches, store it
    string snapshot_filename = Thea::FilePath::concat(search_path, i->path().filename().string());
    THEA_CONSOLE << "Deleting snapshot: " << snapshot_filename;
    std::remove(snapshot_filename.c_str());
  }
}


void MVFCN::loadState(const string& state_filename)
{
  state.clear();
  state.resize(num_camera_orbits + 4, 0.5f);
  ifstream state_file(state_filename);
  if (!state_file.good())
    return;
  for (size_t v = 0; v < state.size(); ++v)
  {    
    state_file >> state[v];
  }
  state_file.close();
}

void MVFCN::saveState(const string& state_filename)
{
  ofstream state_file(state_filename);
  if (!state_file.good())
  {
    THEA_ERROR << "Could not save file: " << state_filename << std::endl;
    return;
  }
  for (size_t v = 0; v < state.size(); ++v)
  {
    state_file << state[v] << " ";
  }
  state_file.close();
}


void MVFCN::get_gpus()
{
  gpus.clear();
  if (Settings::gpu_use == "false")
  {
    return;
  }

  if (Settings::gpu_use == "all")
  {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    for (int i = 0; i < count; ++i)
    {
      gpus.push_back(i);
    }
  }
  else
  {
    vector<string> strings;
    boost::split(strings, Settings::gpu_use, boost::is_any_of(","));
    for (int i = 0; i < strings.size(); ++i)
    {
      gpus.push_back(boost::lexical_cast<int>(strings[i]));
    }
  }
}
#endif


bool MVFCN::createAuxiliaryDirectories(const string& dataset_path, const bool skip_rendering_directories)
{
  boost::filesystem::path dir1(FilePath::concat(dataset_path, PRENDERED_IMAGES_FOLDER));
  boost::filesystem::path dir2(FilePath::concat(dataset_path, DEPTH_IMAGES_FOLDER));
  boost::filesystem::path dir3(FilePath::concat(dataset_path, AUX_IMAGES_FOLDER)); // SDF/UP change
  boost::filesystem::path dir4(FilePath::concat(dataset_path, SEGMENTATION_IMAGES_FOLDER));
  boost::filesystem::path dir5(FilePath::concat(dataset_path, SEGMENTATION_COLOR_IMAGE_FOLDER));
  boost::filesystem::path dir6(FilePath::concat(dataset_path, TRIANGLEID_IMAGES_FOLDER));
  boost::filesystem::path dir7(FilePath::concat(dataset_path, MESH_METADATA_FOLDER)); // rename to RENDERING
  boost::filesystem::path dir8(FilePath::concat(dataset_path, LEARNING_METADATA_FOLDER));
  boost::filesystem::path dir9(FilePath::concat(dataset_path, OUTPUT_MESH_METADATA_FOLDER));
  boost::filesystem::path dir10(FilePath::concat(dataset_path, OUTPUT_SEGMENTANTIONS_FOLDER));

  //if (!skip_rendering_directories)
  //{
  //  boost::filesystem::remove_all(dir1);
  //  boost::filesystem::remove_all(dir2);
  //  boost::filesystem::remove_all(dir3);
  //  boost::filesystem::remove_all(dir4);
  //  boost::filesystem::remove_all(dir5);
  //  boost::filesystem::remove_all(dir6);
  //}
  //boost::filesystem::remove_all(dir7);
  //boost::filesystem::remove_all(dir8);
  //boost::filesystem::remove_all(dir9);

  if (!skip_rendering_directories)
  {
    if (!boost::filesystem::exists(dir1))
    {
      if (!boost::filesystem::create_directory(dir1))
      {
        THEA_ERROR << "Cannot create the necessary rendering directory " << dir1.string() << " in " << dataset_path << "[no write access? invalid dataset path?]";
        return false;
      }
      THEA_CONSOLE << "Created rendering directory: " << std::endl << dir1.string();
    }

    if (!boost::filesystem::exists(dir2))
    {
      if (!boost::filesystem::create_directory(dir2))
      {
        THEA_ERROR << "Cannot create the necessary rendering directory " << dir2.string() << " in " << dataset_path << "[no write access? invalid dataset path?]";
        return false;
      }
      THEA_CONSOLE << "Created rendering directory: " << std::endl << dir2.string();
    }

    if (!boost::filesystem::exists(dir3))
    {
      if (!boost::filesystem::create_directory(dir3))
      {
        THEA_ERROR << "Cannot create the necessary rendering directory " << dir3.string() << " in " << dataset_path << "[no write access? invalid dataset path?]";
        return false;
      }
      THEA_CONSOLE << "Created rendering directory: " << std::endl << dir3.string();
    }

    if (!boost::filesystem::exists(dir4))
    {
      if (!boost::filesystem::create_directory(dir4))
      {
        THEA_ERROR << "Cannot create the necessary rendering directory " << dir4.string() << " in " << dataset_path << "[no write access? invalid dataset path?]";
        return false;
      }
      THEA_CONSOLE << "Created rendering directory: " << std::endl << dir4.string();
    }

    if (!boost::filesystem::exists(dir5))
    {
      if (!boost::filesystem::create_directory(dir5))
      {
        THEA_ERROR << "Cannot create the necessary rendering directory " << dir5.string() << " in " << dataset_path << "[no write access? invalid dataset path?]";
        return false;
      }
      THEA_CONSOLE << "Created rendering directory: " << std::endl << dir5.string();
    }

    if (!boost::filesystem::exists(dir6))
    {
      if (!boost::filesystem::create_directory(dir6))
      {
        THEA_ERROR << "Cannot create the necessary rendering directory " << dir6.string() << " in " << dataset_path << "[no write access? invalid dataset path?]";
        return false;
      }
      THEA_CONSOLE << "Created rendering directory: " << std::endl << dir6.string();
    }


    if (!boost::filesystem::exists(dir7))
    {
      if (!boost::filesystem::create_directory(dir7))
      {
        THEA_ERROR << "Cannot create the necessary mesh metadata directory " << dir7.string() << " in " << dataset_path << "[no write access? invalid dataset path?]";
        return false;
      }
      THEA_CONSOLE << "Created mesh metadata directory: " << std::endl << dir7.string();
    }
  }


  if (!boost::filesystem::exists(dir8))
  {
    if (!boost::filesystem::create_directory(dir8))
    {
      THEA_ERROR << "Cannot create the necessary learning metadata directory " << dir8.string() << " in " << dataset_path << "[no write access? invalid dataset path?]";
      return false;
    }
    THEA_CONSOLE << "Created learning metadata directory: " << std::endl << dir8.string();
  }

  if (!boost::filesystem::exists(dir9))
  {
    if (!boost::filesystem::create_directory(dir9))
    {
      THEA_ERROR << "Cannot create the necessary output metadata directory " << dir9.string() << " in " << dataset_path << "[no write access? invalid dataset path?]";
      return false;
    }
    THEA_CONSOLE << "Created output metadata directory: " << std::endl << dir9.string();
  }

  if (!boost::filesystem::exists(dir10))
  {
    if (!boost::filesystem::create_directory(dir10))
    {
      THEA_ERROR << "Cannot create the necessary output segmentation directory " << dir10.string() << " in " << dataset_path << "[no write access? invalid dataset path?]";
      return false;
    }
    THEA_CONSOLE << "Created output segmentation directory: " << std::endl << dir10.string();
  }

  return true;
}
