# ShapePFCN
Shape Projective Fully Convolutional Network

This is the implementation of the ShapePFCN network described in this paper:
Evangelos Kalogerakis, Melinos Averkiou, Subhransu Maji, Siddhartha Chaudhuri, "3D Shape Segmentation with Projective Convolutional Networks", Proceedings of the IEEE Computer Vision and Pattern Recognition (CVPR) 2017 (oral presentation)

Project page:
http://people.cs.umass.edu/~kalo/papers/shapepfcn/index.html

Arxiv most recent version:
https://arxiv.org/abs/1612.02808

================================

To compile in Linux (we assume 32 threads for compilation, change make's -j32 option accordingly):

a) first compile Siddhartha Chaudhuri's "Thea" library:
   a1) compile Thea's dependencies:
   
       cd TheaDepsUnix/Source/
       ./install-defaults.sh --user <your_user_name> --with-osmesa -j32 
       cd ../../
	   cp -R  TheaDepsUnix/Source/Installations/include/GL TheaDepsUnix/Source/Mesa/mesa-11.0.7/include
	   
	a2) compile Thea library (note: adjust the cuda's dir accordingly)

		cd Thea/Code/Build	
		cmake -DTHEA_INSTALLATIONS_ROOT=../../../TheaDepsUnix/Source/Installations/ -DTHEA_GL_OSMESA=TRUE -DOSMesa_INCLUDE_DIR=../../../TheaDepsUnix/Source/Mesa/mesa-11.0.7/include/ -DOSMesa_GLU_LIBRARIES=../../../TheaDepsUnix/Source/Mesa/mesa-11.0.7/lib -DOPENCL_INCLUDE_DIRS=/usr/local/cuda75/toolkit/7.5.18/include  -DOPENCL_LIBRARIES=/usr/local/cuda75/toolkit/7.5.18/lib64 -DCMAKE_BUILD_TYPE=Release
		make -j32
		cd ../../../
		ln -s Thea/Code/Build/Output/lib lib

b) given that Thea's libraries were compiled successfully (for questions related to Thea, email Siddhartha Chaudhuri),
   the next step is to compile our version of caffe (sorry, we modified caffe to incorporate our own data & projection layers):
      
   cd caffe-ours   
   make -j32
   cd ../
   
   (note: you may need to adjust the library paths in caffe-ours\Makefile.config according to your system, also you will need
   to install the libraries that caffe requires: http://caffe.berkeleyvision.org/installation.html)
   
c) given that caffe was compiled successfully, you can now compile ShapePFCN:

   make -j32
   
   (note: you may need to adjust the library paths in Makefile.config according to your system)
   
   
================================

To run the training procedure (the first command renders images, the second runs the network):

./build_release/mvfcn.bin --skip-testing --do-only-rendering --train-meshes-path  <your_path_to_training_data>
./build_release/mvfcn.bin --skip-testing --skip-train-rendering --train-meshes-path  <your_path_to_training_data> --gpu-use 0

Notes:
=> the gpu-use options specify the id of the GPU in your system, 0 means your first GPU card if you have many)
=> for more options, type: ./build_release/mvfcn.bin --help
=> the implementation assumes a GPU card with 24GB memory (e.g., Tesla M40, Quadro P6000). If you don't have such GPU cars, add the following two arguments for both the above commands:
   '--pretraining-batch-splits 4' (for TitanX, 12GB memory) or '--pretraining-batch-splits 8' (for cards with <=8GB mem)
   *and*
   '--skip-mvfcn'
=> if your shapes have consistent upright (gravity) orientation, PLEASE use the arguments for both the above commands:
   --use-upright-coord
   *and*
   --up-vector 0.0 1.0 0.0  (if y-axis is gravity axis, change according to the upright axis of your dataset)
=> if your shapes have consistent upright & frontfacing orientation (i.e., all shapes are consistently aligned), PLEASE use the arguments for both the above commands:
   --use-consisent-coord
   *and*
   --up-vector 0.0 1.0 0.0  (if y-axis is gravity axis, change according to the upright axis of your dataset)   
   (more discussion about the benefits of upright/consistent orientation, check the arxiv v2 version of our paper)   
=> for faster training, you may consider the option: '--baseline-rendering' for both the above commands, which renders models according to a fixed dodecahedron-based camera setting  (the performance drop is minor)
=> you may want to adjust your LD_LIBRARY_PATH so that all required libraries are accessible e.g., in one of the systems we tried our code, before you run the above commands, we execute:
   LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./caffe-ours/build/lib:/usr/local/hdf5_18/1.8.17/lib:/usr/local/openblas/0.2.18/lib:/usr/local/boost/lib:/usr/local/cuda75/toolkit/7.5.18/lib64:/usr/local/cudnn/5.1/lib64/:/usr/local/apps/cuda-driver/libs/375.20/lib64/
   export LD_LIBRARY_PATH

================================
 
To run the testing procedure:

./build_release/mvfcn.bin --skip-training --do-only-rendering --test-meshes-path  <your_path_to_training_data>
./build_release/mvfcn.bin --skip-training --skip-train-rendering --test-meshes-path  <your_path_to_training_data> --gpu-use 0

Same notes as above apply wrt GPU usage, memory, orientation, and "baseline" rendering

For any questions related to the compilation of caffe-ours and execution of ShapePFCN, please contact kalo@cs.umass.edu

================================

Training/Test data format:

The repository includes the airplanes from L-PSB dataset (http://people.cs.umass.edu/~kalo/papers/LabelMeshes/index.html) as an example of the 
data format ShapePFCN supports. There are two possible formats:
a) OBJ files where each training part is stored as a group in an OBJ file (see psbAirplane1 folder and https://en.wikipedia.org/wiki/Wavefront_.obj_file)
b) OFF files where labels are stored in separate label txt files (see psbAirplane2 folder).
   In this case, each OFF has a txt file that ends with the string "_labels.txt". Each pair of lines in these txt files contains a label identifier and a list of integers that are indices to faces having that label (the first face has index 1, not 0).

For testing, no OBJ groups or labels txt files are needed. If they are found in the test directory, they will be simply used for evaluating test accuracy.




   