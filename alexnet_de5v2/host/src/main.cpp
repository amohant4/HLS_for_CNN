// Copyright (C) 2013-2015 Altera Corporation, San Jose, California, USA. All rights reserved. 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this 
// software and associated documentation files (the "Software"), to deal in the Software 
// without restriction, including without limitation the rights to use, copy, modify, merge, 
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to 
// whom the Software is furnished to do so, subject to the following conditions: 
// The above copyright notice and this permission notice shall be included in all copies or 
// substantial portions of the Software. 
//  
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
// OTHER DEALINGS IN THE SOFTWARE. 
//  
// This agreement shall be governed in all respects by the laws of the State of California and 
// by the laws of the United States of America. 

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <Windows.h>
#include <stdio.h>
#include <stdlib.h>
#include "CL/opencl.h"
#include "AOCL_Utils.h"
#include "alexnet.h"

using namespace aocl_utils;
using namespace cv;
using namespace std;

// OpenCL runtime configuration
cl_platform_id platform = NULL;
unsigned num_devices = 0;
scoped_array<cl_device_id> device; 
cl_context context = NULL;
scoped_array<cl_command_queue> queue; 
cl_program program = NULL;
scoped_array<cl_kernel> kernel;
scoped_array<cl_mem> in_data_buf, conv1_in_buf, conv1_wt_buf, conv1_bias_buf, conv1_out_buf;
scoped_array<cl_mem> norm1_out_buf, pool1_out_buf; 
scoped_array<cl_mem> conv2_in_buf, conv2_wt_buf, conv2_bias_buf, conv2_out_buf; 
scoped_array<cl_mem> norm2_out_buf, pool2_out_buf; 
scoped_array<cl_mem> conv3_in_buf, conv3_wt_buf, conv3_bias_buf, conv3_out_buf; 
scoped_array<cl_mem> conv4_in_buf, conv4_wt_buf, conv4_bias_buf, conv4_out_buf; 
scoped_array<cl_mem> conv5_in_buf, conv5_wt_buf, conv5_bias_buf, conv5_out_buf; 
scoped_array<cl_mem> pool5_out_buf; 
scoped_array<cl_mem> fc6_wt_buf, fc6_bias_buf, fc6_out_buf;
scoped_array<cl_mem> fc7_wt_buf, fc7_bias_buf, fc7_out_buf;
scoped_array<cl_mem> fc8_wt_buf, fc8_bias_buf, fc8_out_buf;
scoped_array<cl_mem> label_out_buf;
const unsigned num_kernels=5;

// Problem data.
const unsigned batch_size = 50;
const unsigned test_size = 5000;
unsigned image_no=0;
unsigned batch=0;
char aocx_name[40] = "alexnet_de5v2";
unsigned actual_batch_size = batch_size;
unsigned actual_test_size = test_size;
bool display=0;
bool measure_power=0;
bool print_time=0;
char im_mean[3]={104,117,123};
char label_name[1000][40];
double batch_run_time=1.0;

const unsigned IM_W = 227;	// input image size
const unsigned no_ch = 3;
const unsigned N_group1 = 1;
const unsigned N_conv1 = 96; // Number of convolution outputs
const unsigned K_conv1 = 11;  
const unsigned S_conv1 = 4;  
const unsigned P_conv1 = 0;
const unsigned conv1_out_dim = (IM_W+2*P_conv1-K_conv1)/S_conv1+1;
const unsigned matrix_dim1 = no_ch/N_group1*K_conv1*K_conv1;
const unsigned N_elem1 = (matrix_dim1 % CONV_BLOCK_SIZE ==0)? matrix_dim1 : matrix_dim1 + (CONV_BLOCK_SIZE-(matrix_dim1 % CONV_BLOCK_SIZE));	// Padding the matrix width to make it a multiple of CONV_BLOCK_SIZE
const unsigned conv1_out_pad = ((conv1_out_dim*conv1_out_dim)%CONV_BLOCK_SIZE==0) ? (conv1_out_dim*conv1_out_dim) : (conv1_out_dim*conv1_out_dim) + (CONV_BLOCK_SIZE - ((conv1_out_dim*conv1_out_dim)%CONV_BLOCK_SIZE));
const unsigned N_conv1_pad = (N_conv1%CONV_BLOCK_SIZE==0)? N_conv1: N_conv1 + (CONV_BLOCK_SIZE - (N_conv1%CONV_BLOCK_SIZE));

const unsigned K_pool1 = 3;
const unsigned S_pool1 = 2;
const unsigned pool1_out_dim = (conv1_out_dim-K_pool1)/S_pool1+1;

const unsigned N_group2 = 2;
const unsigned N_conv2 = 256;
const unsigned K_conv2 = 5;
const unsigned S_conv2 = 1;
const unsigned P_conv2 = 2;
const unsigned conv2_out_dim = (pool1_out_dim+2*P_conv2-K_conv2)/S_conv2+1;
const unsigned matrix_dim2 = N_conv1/N_group2*K_conv2*K_conv2;
const unsigned N_elem2 = (matrix_dim2 % CONV_BLOCK_SIZE ==0)? matrix_dim2 : matrix_dim2 + (CONV_BLOCK_SIZE-(matrix_dim2 % CONV_BLOCK_SIZE));	// Padding the matrix width to make it a multiple of CONV_BLOCK_SIZE
const unsigned conv2_out_pad = ((conv2_out_dim*conv2_out_dim)%CONV_BLOCK_SIZE==0) ? (conv2_out_dim*conv2_out_dim) : (conv2_out_dim*conv2_out_dim) + (CONV_BLOCK_SIZE - ((conv2_out_dim*conv2_out_dim)%CONV_BLOCK_SIZE));
const unsigned N_conv2_pad = (N_conv2%CONV_BLOCK_SIZE==0)? N_conv2: N_conv2 + (CONV_BLOCK_SIZE - (N_conv2%CONV_BLOCK_SIZE));

const unsigned K_pool2 = 3;
const unsigned S_pool2 = 2;
const unsigned pool2_out_dim = (conv2_out_dim-K_pool2)/S_pool2+1;

const unsigned N_group3 = 1;
const unsigned N_conv3 = 384;
const unsigned K_conv3 = 3;
const unsigned S_conv3 = 1;
const unsigned P_conv3 = 1;
const unsigned conv3_out_dim = (pool2_out_dim+2*P_conv3-K_conv3)/S_conv3+1;
const unsigned matrix_dim3 = N_conv2/N_group3*K_conv3*K_conv3;
const unsigned N_elem3 = (matrix_dim3 % CONV_BLOCK_SIZE ==0)? matrix_dim3 : matrix_dim3 + (CONV_BLOCK_SIZE-(matrix_dim3 % CONV_BLOCK_SIZE));	// Padding the matrix width to make it a multiple of CONV_BLOCK_SIZE
const unsigned conv3_out_pad = ((conv3_out_dim*conv3_out_dim)%CONV_BLOCK_SIZE==0) ? (conv3_out_dim*conv3_out_dim) : (conv3_out_dim*conv3_out_dim) + (CONV_BLOCK_SIZE - ((conv3_out_dim*conv3_out_dim)%CONV_BLOCK_SIZE));
const unsigned N_conv3_pad = (N_conv3%CONV_BLOCK_SIZE==0)? N_conv3: N_conv3 + (CONV_BLOCK_SIZE - (N_conv3%CONV_BLOCK_SIZE));

const unsigned N_group4 = 2;
const unsigned N_conv4 = 384;
const unsigned K_conv4 = 3;
const unsigned S_conv4 = 1;
const unsigned P_conv4 = 1;
const unsigned conv4_out_dim = (conv3_out_dim+2*P_conv4-K_conv4)/S_conv4+1;
const unsigned matrix_dim4 = N_conv3/N_group4*K_conv4*K_conv4;
const unsigned N_elem4 = (matrix_dim4 % CONV_BLOCK_SIZE ==0)? matrix_dim4 : matrix_dim4 + (CONV_BLOCK_SIZE-(matrix_dim4 % CONV_BLOCK_SIZE));	// Padding the matrix width to make it a multiple of CONV_BLOCK_SIZE
const unsigned conv4_out_pad = ((conv4_out_dim*conv4_out_dim)%CONV_BLOCK_SIZE==0) ? (conv4_out_dim*conv4_out_dim) : (conv4_out_dim*conv4_out_dim) + (CONV_BLOCK_SIZE - ((conv4_out_dim*conv4_out_dim)%CONV_BLOCK_SIZE));
const unsigned N_conv4_pad = ((N_conv4/N_group4)%CONV_BLOCK_SIZE==0)? N_conv4: N_conv4 + N_group4*(CONV_BLOCK_SIZE - ((N_conv4/N_group4)%CONV_BLOCK_SIZE));	// special case for CONV_BLOCK_SIZE=128

const unsigned N_group5 = 2;
const unsigned N_conv5 = 256;
const unsigned K_conv5 = 3;
const unsigned S_conv5 = 1;
const unsigned P_conv5 = 1;
const unsigned conv5_out_dim = (conv4_out_dim+2*P_conv5-K_conv5)/S_conv5+1;
const unsigned matrix_dim5 = N_conv4/N_group5*K_conv5*K_conv5;
const unsigned N_elem5 = (matrix_dim5 % CONV_BLOCK_SIZE ==0)? matrix_dim5 : matrix_dim5 + (CONV_BLOCK_SIZE-(matrix_dim5 % CONV_BLOCK_SIZE));	// Padding the matrix width to make it a multiple of CONV_BLOCK_SIZE
const unsigned conv5_out_pad = ((conv5_out_dim*conv5_out_dim)%CONV_BLOCK_SIZE==0) ? (conv5_out_dim*conv5_out_dim) : (conv5_out_dim*conv5_out_dim) + (CONV_BLOCK_SIZE - ((conv5_out_dim*conv5_out_dim)%CONV_BLOCK_SIZE));
const unsigned N_conv5_pad = (N_conv5%CONV_BLOCK_SIZE==0)? N_conv5: N_conv5 + (CONV_BLOCK_SIZE - (N_conv5%CONV_BLOCK_SIZE));

const unsigned K_pool5 = 3; 
const unsigned S_pool5 = 2;
const unsigned pool5_out_dim = (conv5_out_dim-K_pool5)/S_pool5+1;

const unsigned fc6_out_dim = 4096;
const unsigned fc6_wt_dim = (fc6_out_dim*N_conv5*pool5_out_dim*pool5_out_dim);
const unsigned fc6_in_dim = (N_conv5*pool5_out_dim*pool5_out_dim);

const unsigned fc7_out_dim = 4096;
const unsigned fc7_wt_dim = (fc7_out_dim*fc6_out_dim);

const unsigned fc8_out_dim = 1000;
const unsigned fc8_wt_dim = (fc8_out_dim*fc7_out_dim);

//scoped_array<short int> in_data[1];
scoped_array<scoped_aligned_ptr<short int> > in_data;
scoped_array<scoped_aligned_ptr<char> > conv1_wt, conv1_bias;
scoped_array<scoped_aligned_ptr<unsigned char> > in_image;
scoped_array<scoped_aligned_ptr<short int> > conv1_in, conv1_out, norm1_out, pool1_out;
scoped_array<scoped_aligned_ptr<char> > conv2_wt, conv2_bias;
scoped_array<scoped_aligned_ptr<short int> > conv2_in, conv2_out, norm2_out, pool2_out;
scoped_array<scoped_aligned_ptr<char> > conv3_wt, conv3_bias;
scoped_array<scoped_aligned_ptr<short int> > conv3_in, conv3_out;
scoped_array<scoped_aligned_ptr<char> > conv4_wt, conv4_bias;
scoped_array<scoped_aligned_ptr<short int> > conv4_in, conv4_out;
scoped_array<scoped_aligned_ptr<char> > conv5_wt, conv5_bias;
scoped_array<scoped_aligned_ptr<short int> > conv5_in, conv5_out, pool5_out;
scoped_array<scoped_aligned_ptr<char> > fc6_wt, fc6_bias, fc7_wt, fc7_bias, fc8_wt, fc8_bias;
scoped_array<scoped_aligned_ptr<short int> > fc6_out, fc7_out, fc8_out, label_out, temp_label_out;

scoped_array<scoped_array<int> > ref_conv1_out, ref_norm1_out, ref_pool1_out;
scoped_array<scoped_array<int> > ref_conv2_out, ref_norm2_out, ref_pool2_out;
scoped_array<scoped_array<int> > ref_conv3_out;
scoped_array<scoped_array<int> > ref_conv4_out;
scoped_array<scoped_array<int> > ref_conv5_out, ref_pool5_out;
scoped_array<scoped_array<int> > ref_fc6_out, ref_fc7_out, ref_fc8_out;
scoped_array<scoped_array<unsigned short> > ref_label_out;

char input_data[] = "../Data/indata.bin";
char conv1_wt_file[] = "../Model/AlexNet/conv1_wt.txt";
char conv1_bias_file[] = "../Model/AlexNet/conv1_bias.txt";
char conv1_out_file[] = "../Model/AlexNet/conv1_out.txt";
char norm1_out_file[] = "../Model/AlexNet/norm1_out.txt";
char pool1_out_file[] = "../Model/AlexNet/pool1_out.txt";
char conv2_wt_file[] = "../Model/AlexNet/conv2_wt.txt";
char conv2_bias_file[] = "../Model/AlexNet/conv2_bias.txt";
char conv2_out_file[] = "../Model/AlexNet/conv2_out.txt";
char norm2_out_file[] = "../Model/AlexNet/norm2_out.txt";
char pool2_out_file[] = "../Model/AlexNet/pool2_out.txt";
char conv3_wt_file[] = "../Model/AlexNet/conv3_wt.txt";
char conv3_bias_file[] = "../Model/AlexNet/conv3_bias.txt";
char conv3_out_file[] = "../Model/AlexNet/conv3_out.txt";
char conv4_wt_file[] = "../Model/AlexNet/conv4_wt.txt";
char conv4_bias_file[] = "../Model/AlexNet/conv4_bias.txt";
char conv4_out_file[] = "../Model/AlexNet/conv4_out.txt";
char conv5_wt_file[] = "../Model/AlexNet/conv5_wt.txt";
char conv5_bias_file[] = "../Model/AlexNet/conv5_bias.txt";
char conv5_out_file[] = "../Model/AlexNet/conv5_out.txt";
char pool5_out_file[] = "../Model/AlexNet/pool5_out.txt";
char fc6_wt_file[] = "../Model/AlexNet/fc6_wt.bin";
char fc6_bias_file[] = "../Model/AlexNet/fc6_bias.txt";
char fc6_out_file[] = "../Model/AlexNet/fc6_out.txt";
char fc7_wt_file[] = "../Model/AlexNet/fc7_wt.bin";
char fc7_bias_file[] = "../Model/AlexNet/fc7_bias.txt";
char fc7_out_file[] = "../Model/AlexNet/fc7_out.txt";
char fc8_wt_file[] = "../Model/AlexNet/fc8_wt.bin";
char fc8_bias_file[] = "../Model/AlexNet/fc8_bias.txt";
char fc8_out_file[] = "../Model/AlexNet/fc8_out.txt";
char inlabel_file[] = "../Data/inlabel.bin";
char inlabelwords_file[] = "../Data/synset_data.txt";

// Function prototypes
bool init_opencl();
void init_problem();
void run();
void cleanup();

DWORD WINAPI display_thread(LPVOID lpParameter)
{
    Mat image(IM_W,IM_W,CV_8UC3);
	Mat scaled_image;
	Size new_size(2*IM_W,2*IM_W);
	Sleep(1000);
	while(1)
	{

	for(int i=0;i<IM_W;i++)	//row
	{
		for(int j=0;j<IM_W;j++)	//column
		{
			Vec3b mypixel = image.at<Vec3b>(Point(j,i));
			mypixel[0]=in_image[0][i*IM_W + j];
			mypixel[1]=in_image[0][IM_W*IM_W + i*IM_W + j];
			mypixel[2]=in_image[0][2*IM_W*IM_W + i*IM_W + j];
			image.at<Vec3b>(Point(j,i)) = mypixel;
		}
	}
	resize(image,scaled_image,new_size);
    Mat image2;
	Scalar pixel = Scalar(255,255,255);
	copyMakeBorder( scaled_image, image2, 0, 0, 0, 300, BORDER_CONSTANT, pixel );
	
	double fps = actual_batch_size/batch_run_time;
	char fps_text[30];
	sprintf(fps_text,"Processing speed: %.2f FPS",fps);
	
	Point textOrg1(460,50);
	Point textOrg2(460,100);
	Point textOrg3(460,150);
	Point textOrg4(460,200);
	Point textOrg5(460,250);
	Point textOrg6(460,350);
	Point textOrg7(460,420);
	putText(image2, label_name[temp_label_out[0][0]], textOrg1, FONT_HERSHEY_DUPLEX, 0.6, Scalar(0,0,0), 1, 8);
	putText(image2, label_name[temp_label_out[0][1]], textOrg2, FONT_HERSHEY_DUPLEX, 0.6, Scalar::all(0), 1, 8);
	putText(image2, label_name[temp_label_out[0][2]], textOrg3, FONT_HERSHEY_DUPLEX, 0.6, Scalar::all(0), 1, 8);
	putText(image2, label_name[temp_label_out[0][3]], textOrg4, FONT_HERSHEY_DUPLEX, 0.6, Scalar::all(0), 1, 8);
	putText(image2, label_name[temp_label_out[0][4]], textOrg5, FONT_HERSHEY_DUPLEX, 0.6, Scalar::all(0), 1, 8);
	bool found=0;
	for(int i=0;i<TOP_K;i++)
	{
		if(ref_label_out[0][batch*actual_batch_size]==temp_label_out[0][i])
			found=1;
	}
	Scalar color = (found==1) ? Scalar(0,255,0) : Scalar(0,0,255);
	putText(image2, label_name[ref_label_out[0][batch*actual_batch_size]], textOrg6, FONT_HERSHEY_DUPLEX, 0.6, color, 1, 8);
	putText(image2, fps_text, textOrg7, FONT_HERSHEY_DUPLEX, 0.6, Scalar(0,0,0), 1, 8);
		
	imshow( "Running Alexnet", image2 );                  // Show our image inside it.
	waitKey(25);                                         // Wait for a keystroke in the window 
	
    }
	
	return 0;
}

// Entry point.
int main( int argc, char** argv ) {

	//parse command line arguments
	for(int i=1;i<argc;i++)
	{
		if(strcmp(argv[i],"-display")==0)
			display=1;
		else if(strcmp(argv[i],"-power")==0)
			measure_power=1;
		else if(strcmp(argv[i],"-time")==0)
			print_time=1;
		else if(strcmp(argv[i],"-batch")==0)
			actual_batch_size=atoi(argv[i+1]);
		else if(strcmp(argv[i],"-test")==0)
			actual_test_size=atoi(argv[i+1]);
		else if(strcmp(argv[i],"-aocx")==0)
			strcpy(aocx_name,argv[i+1]);
	}

  if(display==0)
	  printf("Image and predicitons will not be displayed\n");
  else
	  printf("Image and predictions will be displayed\n");
  if(measure_power==0)
	  printf("Measured power will not be displayed\n");
  else
	  printf("Measured power will be displayed\n");
  if(print_time==0)
	  printf("Kernel time will not be displayed\n");
  else
	  printf("Kernel time will be displayed\n");

  printf("\n");
  // Initialize OpenCL.
  if(!init_opencl()) {
    return -1;
  }

  // Initialize the problem data.
  init_problem();
  int i;
  DWORD myThreadID;
  HANDLE myHandle;
  if(display==1)
	myHandle = CreateThread(0, 0, display_thread, &i, 0, &myThreadID);
  
  // Run the kernel.
  run();
  
  if(display==1)
	CloseHandle(myHandle);
  
  // Free the resources allocated
  cleanup();
  
  return 0;
}

/////// HELPER FUNCTIONS ///////

// Initializes the OpenCL objects.
bool init_opencl() {
  cl_int status;

  printf("Initializing OpenCL\n");

  if(!setCwdToExeDir()) {
    return false;
  }

  // Get the OpenCL platform.
  platform = findPlatform("Altera");
  if(platform == NULL) {
    printf("ERROR: Unable to find Altera OpenCL platform.\n");
    return false;
  }

  // Query the available OpenCL device.
  device.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
  printf("Platform: %s\n", getPlatformName(platform).c_str());
  printf("Using %d device(s)\n", num_devices);
  for(unsigned i = 0; i < num_devices; ++i) {
    printf("  %s\n", getDeviceName(device[i]).c_str());
  }

  // Create the context.
  context = clCreateContext(NULL, num_devices, device, NULL, NULL, &status);
  checkError(status, "Failed to create context");

  // Create the program for all device. Use the first device as the
  // representative device (assuming all device are of the same type).
  std::string binary_file = getBoardBinaryFile(aocx_name, device[0]);
  printf("Using AOCX: %s\n", binary_file.c_str());
  program = createProgramFromBinary(context, binary_file.c_str(), device, num_devices);

  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

  // Create per-device objects.
  
  queue.reset(num_kernels);
  kernel.reset(num_kernels);
  
  in_data_buf.reset(num_devices);
  conv1_wt_buf.reset(num_devices);
  conv1_bias_buf.reset(num_devices);
  conv1_in_buf.reset(num_devices);
  conv1_out_buf.reset(num_devices);
  
  norm1_out_buf.reset(num_devices);
  pool1_out_buf.reset(num_devices);
  
  conv2_wt_buf.reset(num_devices);
  conv2_bias_buf.reset(num_devices);
  conv2_in_buf.reset(num_devices);
  conv2_out_buf.reset(num_devices);
  
  norm2_out_buf.reset(num_devices);
  pool2_out_buf.reset(num_devices);

  conv3_wt_buf.reset(num_devices);
  conv3_bias_buf.reset(num_devices);
  conv3_in_buf.reset(num_devices);
  conv3_out_buf.reset(num_devices);

  conv4_wt_buf.reset(num_devices);
  conv4_bias_buf.reset(num_devices);
  conv4_in_buf.reset(num_devices);
  conv4_out_buf.reset(num_devices);

  conv5_wt_buf.reset(num_devices);
  conv5_bias_buf.reset(num_devices);
  conv5_in_buf.reset(num_devices);
  conv5_out_buf.reset(num_devices);
  
  pool5_out_buf.reset(num_devices);

  fc6_wt_buf.reset(num_devices);
  fc6_bias_buf.reset(num_devices);
  fc6_out_buf.reset(num_devices);
  
  fc7_wt_buf.reset(num_devices);
  fc7_bias_buf.reset(num_devices);
  fc7_out_buf.reset(num_devices);

  fc8_wt_buf.reset(num_devices);
  fc8_bias_buf.reset(num_devices);
  fc8_out_buf.reset(num_devices);

  label_out_buf.reset(test_size);
  
    // Command queue.
  for(int i=0;i<num_kernels;i++)
  {
    queue[i] = clCreateCommandQueue(context, device[0], CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue");
  }

	// Kernels

	kernel[0] = clCreateKernel(program, "conv", &status);
    checkError(status, "Failed to create conv kernel");

	kernel[1] = clCreateKernel(program, "norm_cross", &status);
    checkError(status, "Failed to create norm_cross kernel");

	kernel[2] = clCreateKernel(program, "max_pool", &status);
    checkError(status, "Failed to create max_pool kernel");

	kernel[3] = clCreateKernel(program, "innerproduct", &status);
    checkError(status, "Failed to create innerproduct kernel");

	kernel[4] = clCreateKernel(program, "predict_label", &status);
    checkError(status, "Failed to create prdict_label kernel");

	// Input buffers.
    in_data_buf[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_1_ALTERA, 
        IM_W*IM_W*no_ch*batch_size* sizeof(short int), NULL, &status);
    checkError(status, "Failed to create buffer for input image");

    conv1_wt_buf[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_1_ALTERA, 
    	N_elem1*N_conv1_pad * sizeof(char), NULL, &status);
    checkError(status, "Failed to create buffer for convolution weights");
	conv1_bias_buf[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_1_ALTERA, 
        N_conv1_pad * sizeof(char), NULL, &status);
    checkError(status, "Failed to create buffer for convolution bias");

	conv2_wt_buf[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_1_ALTERA, 
    	N_elem2*N_conv2_pad * sizeof(char), NULL, &status);
    checkError(status, "Failed to create buffer for convolution weights");
	conv2_bias_buf[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_1_ALTERA, 
        N_conv2_pad * sizeof(char), NULL, &status);
    checkError(status, "Failed to create buffer for convolution bias");

	conv3_wt_buf[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_1_ALTERA, 
    	N_elem3*N_conv3_pad * sizeof(char), NULL, &status);
    checkError(status, "Failed to create buffer for convolution weights");
	conv3_bias_buf[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_1_ALTERA, 
        N_conv3_pad * sizeof(char), NULL, &status);
    checkError(status, "Failed to create buffer for convolution bias");

	conv4_wt_buf[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_1_ALTERA, 
    	N_elem4*N_conv4_pad * sizeof(char), NULL, &status);
    checkError(status, "Failed to create buffer for convolution weights");
	conv4_bias_buf[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_1_ALTERA, 
        N_conv4_pad * sizeof(char), NULL, &status);
    checkError(status, "Failed to create buffer for convolution bias");

	conv5_wt_buf[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_1_ALTERA, 
    	N_elem5*N_conv5_pad * sizeof(char), NULL, &status);
    checkError(status, "Failed to create buffer for convolution weights");
	conv5_bias_buf[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_1_ALTERA, 
        N_conv5_pad * sizeof(char), NULL, &status);
    checkError(status, "Failed to create buffer for convolution bias");

	fc6_wt_buf[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_1_ALTERA, 
        fc6_wt_dim * sizeof(char), NULL, &status);
    checkError(status, "Failed to create buffer for innerproduct weights");
	fc6_bias_buf[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_1_ALTERA, 
        fc6_out_dim * sizeof(char), NULL, &status);
    checkError(status, "Failed to create buffer for innerproduct bias");
	
	fc7_wt_buf[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_1_ALTERA, 
        fc7_wt_dim * sizeof(char), NULL, &status);
    checkError(status, "Failed to create buffer for innerproduct weights");
	fc7_bias_buf[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_1_ALTERA, 
        fc7_out_dim * sizeof(char), NULL, &status);
    checkError(status, "Failed to create buffer for innerproduct bias");

	fc8_wt_buf[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_1_ALTERA, 
        fc8_wt_dim * sizeof(char), NULL, &status);
    checkError(status, "Failed to create buffer for innerproduct weights");
	fc8_bias_buf[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_BANK_1_ALTERA, 
        fc8_out_dim * sizeof(char), NULL, &status);
    checkError(status, "Failed to create buffer for innerproduct bias");

	// Output buffer.
    conv1_in_buf[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_BANK_1_ALTERA, 
		N_group1*N_elem1*conv1_out_pad * sizeof(short int), NULL, &status);
    conv1_out_buf[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_BANK_1_ALTERA,
    	conv1_out_pad*N_conv1_pad * sizeof(short int), NULL, &status);
    norm1_out_buf[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_BANK_1_ALTERA,
    	conv1_out_dim*conv1_out_dim*N_conv1 * sizeof(short int), NULL, &status);
    pool1_out_buf[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_BANK_1_ALTERA,
    	pool1_out_dim*pool1_out_dim*N_conv1 * sizeof(short int), NULL, &status);

    conv2_in_buf[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_BANK_1_ALTERA, 
		N_group2*N_elem2*conv2_out_pad * sizeof(short int), NULL, &status);
    conv2_out_buf[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_BANK_1_ALTERA,
    	conv2_out_pad*N_conv2_pad * sizeof(short int), NULL, &status);
    norm2_out_buf[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_BANK_1_ALTERA,
    	conv2_out_dim*conv2_out_dim*N_conv2 * sizeof(short int), NULL, &status);
    pool2_out_buf[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_BANK_1_ALTERA,
    	pool2_out_dim*pool2_out_dim*N_conv2 * sizeof(short int), NULL, &status);

    conv3_in_buf[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_BANK_1_ALTERA, 
		N_group3*N_elem3*conv3_out_pad * sizeof(short int), NULL, &status);
    conv3_out_buf[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_BANK_1_ALTERA,
    	conv3_out_pad*N_conv3_pad * sizeof(short int), NULL, &status);
    
    conv4_in_buf[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_BANK_1_ALTERA, 
		N_group4*N_elem4*conv4_out_pad * sizeof(short int), NULL, &status);
    conv4_out_buf[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_BANK_1_ALTERA,
    	conv4_out_pad*N_conv4_pad * sizeof(short int), NULL, &status);
    
    conv5_in_buf[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_BANK_1_ALTERA, 
		N_group5*N_elem5*conv5_out_pad * sizeof(short int), NULL, &status);
    conv5_out_buf[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_BANK_1_ALTERA,
    	conv5_out_pad*N_conv5_pad * sizeof(short int), NULL, &status);
    pool5_out_buf[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_BANK_1_ALTERA,
    	pool5_out_dim*pool5_out_dim*N_conv5 * sizeof(short int), NULL, &status);
    
	fc6_out_buf[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_BANK_1_ALTERA,
    	fc6_out_dim * sizeof(short int), NULL, &status);
	fc7_out_buf[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_BANK_1_ALTERA,
    	fc7_out_dim * sizeof(short int), NULL, &status);
	fc8_out_buf[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_BANK_1_ALTERA,
    	fc8_out_dim * sizeof(short int), NULL, &status);

	for(int i=0;i<actual_test_size;i++)
	{
		label_out_buf[i] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_BANK_1_ALTERA,
			TOP_K * sizeof(short int), NULL, &status);
	}
    printf("Buffer creation successful\n");
	return true;
}

// Initialize the data for the problem. Requires num_devices to be known.
void init_problem() {
  if(num_devices == 0) {
    checkError(-1, "No devices");
  }
  
  printf("Reading Labels from %s\n",inlabelwords_file);
 
  FILE* inlabel = fopen(inlabelwords_file,"r");
	int i=0;
	char line[100];
	while(fgets(line, sizeof(line), inlabel))
	{
			strtok(line,"\n");
			strcpy(label_name[i],line);
			//printf("%s-%s\n",line,label_name[i]);
			i++;
	}
	fclose(inlabel);
 

  printf("Initializing data\n");

  in_data.reset(batch_size);
  in_image.reset(batch_size);
  conv1_wt.reset(num_devices);
  conv1_bias.reset(num_devices);
  conv1_in.reset(num_devices);
  conv1_out.reset(num_devices);
  ref_conv1_out.reset(num_devices);
  norm1_out.reset(num_devices);
  ref_norm1_out.reset(num_devices);
  pool1_out.reset(num_devices);
  ref_pool1_out.reset(num_devices);

  conv2_wt.reset(num_devices);
  conv2_bias.reset(num_devices);
  conv2_in.reset(num_devices);
  conv2_out.reset(num_devices);
  ref_conv2_out.reset(num_devices);
  norm2_out.reset(num_devices);
  ref_norm2_out.reset(num_devices);
  pool2_out.reset(num_devices);
  ref_pool2_out.reset(num_devices);
  
  conv3_wt.reset(num_devices);
  conv3_bias.reset(num_devices);
  conv3_in.reset(num_devices);
  conv3_out.reset(num_devices);
  ref_conv3_out.reset(num_devices);
  
  conv4_wt.reset(num_devices);
  conv4_bias.reset(num_devices);
  conv4_in.reset(num_devices);
  conv4_out.reset(num_devices);
  ref_conv4_out.reset(num_devices);
  
  conv5_wt.reset(num_devices);
  conv5_bias.reset(num_devices);
  conv5_in.reset(num_devices);
  conv5_out.reset(num_devices);
  ref_conv5_out.reset(num_devices);
  pool5_out.reset(num_devices);
  ref_pool5_out.reset(num_devices);
  
  fc6_wt.reset(num_devices);
  fc6_bias.reset(num_devices);
  fc6_out.reset(num_devices);
  ref_fc6_out.reset(num_devices);
  
  fc7_wt.reset(num_devices);
  fc7_bias.reset(num_devices);
  fc7_out.reset(num_devices);
  ref_fc7_out.reset(num_devices);
  
  fc8_wt.reset(num_devices);
  fc8_bias.reset(num_devices);
  fc8_out.reset(num_devices);
  ref_fc8_out.reset(num_devices);

  label_out.reset(test_size);
  temp_label_out.reset(batch_size);
  ref_label_out.reset(num_devices);

  // We create separate arrays for each device so that each device has an aligned buffer. 
	//for(int i=0;i<test_size/batch_size;i++)
	for(int i=0;i<1;i++)
	{
		in_data[i].reset(IM_W*IM_W*no_ch*batch_size);
    	in_image[i].reset(IM_W*IM_W*no_ch*batch_size);
	}
    conv1_wt[0].reset(N_elem1*N_conv1_pad);
    conv1_bias[0].reset(N_conv1_pad);
    conv1_in[0].reset(N_group1*N_elem1*conv1_out_pad);
	conv1_out[0].reset(conv1_out_pad*N_conv1_pad);
    ref_conv1_out[0].reset(conv1_out_dim*conv1_out_dim*N_conv1);
	norm1_out[0].reset(N_conv1*conv1_out_dim*conv1_out_dim);
	ref_norm1_out[0].reset(N_conv1*conv1_out_dim*conv1_out_dim);
	pool1_out[0].reset(N_conv1*pool1_out_dim*pool1_out_dim);
	ref_pool1_out[0].reset(N_conv1*pool1_out_dim*pool1_out_dim);
    
	conv2_wt[0].reset(N_elem2*N_conv2_pad);
    conv2_bias[0].reset(N_conv2_pad);
    conv2_in[0].reset(N_group2*N_elem2*conv2_out_pad);
	conv2_out[0].reset(conv2_out_pad*N_conv2_pad);
    ref_conv2_out[0].reset(conv2_out_dim*conv2_out_dim*N_conv2);
	norm2_out[0].reset(N_conv2*conv2_out_dim*conv2_out_dim);
	ref_norm2_out[0].reset(N_conv2*conv2_out_dim*conv2_out_dim);
	pool2_out[0].reset(N_conv2*pool2_out_dim*pool2_out_dim);
	ref_pool2_out[0].reset(N_conv2*pool2_out_dim*pool2_out_dim);

	conv3_wt[0].reset(N_elem3*N_conv3_pad);
    conv3_bias[0].reset(N_conv3_pad);
    conv3_in[0].reset(N_group3*N_elem3*conv3_out_pad);
	conv3_out[0].reset(conv3_out_pad*N_conv3_pad);
    ref_conv3_out[0].reset(conv3_out_dim*conv3_out_dim*N_conv3);
    
	conv4_wt[0].reset(N_elem4*N_conv4_pad);
    conv4_bias[0].reset(N_conv4_pad);
    conv4_in[0].reset(N_group4*N_elem4*conv4_out_pad);
	conv4_out[0].reset(conv4_out_pad*N_conv4_pad);
    ref_conv4_out[0].reset(conv4_out_dim*conv4_out_dim*N_conv4);
    
	conv5_wt[0].reset(N_elem5*N_conv5_pad);
    conv5_bias[0].reset(N_conv5_pad);
    conv5_in[0].reset(N_group5*N_elem5*conv5_out_pad);
	conv5_out[0].reset(conv5_out_pad*N_conv5_pad);
    ref_conv5_out[0].reset(conv5_out_dim*conv5_out_dim*N_conv5);
	pool5_out[0].reset(N_conv5*pool5_out_dim*pool5_out_dim);
	ref_pool5_out[0].reset(N_conv5*pool5_out_dim*pool5_out_dim);
    
	fc6_wt[0].reset(fc6_wt_dim);
    fc6_bias[0].reset(fc6_out_dim);
    fc6_out[0].reset(fc6_out_dim);
    ref_fc6_out[0].reset(fc6_out_dim);

	fc7_wt[0].reset(fc7_wt_dim);
    fc7_bias[0].reset(fc7_out_dim);
    fc7_out[0].reset(fc7_out_dim);
    ref_fc7_out[0].reset(fc7_out_dim);

	fc8_wt[0].reset(fc8_wt_dim);
    fc8_bias[0].reset(fc8_out_dim);
    fc8_out[0].reset(fc8_out_dim);
    ref_fc8_out[0].reset(fc8_out_dim);
	for(unsigned i=0;i<test_size;i++)
		label_out[i].reset(TOP_K);
	for(unsigned i=0;i<batch_size;i++)
		temp_label_out[i].reset(TOP_K);

    ref_label_out[0].reset(test_size);

  printf("Reading convolution weights from %s\n",conv1_wt_file);
  FILE* inkernel=fopen(conv1_wt_file,"r");
  unsigned k=0;
  int wt;
  for(unsigned i=0;i<N_conv1_pad;i++){
	  for(unsigned j=0;j<N_elem1;j++){
		  if(j<K_conv1*K_conv1*no_ch/N_group1 && i<N_conv1)
		  {
			  fscanf(inkernel,"%d",&wt);
			  conv1_wt[0][k++]=(char)wt;
		  }
		  else // Padding to make it multiple of CONV_BLOCK_SIZE
			  conv1_wt[0][k++]=0;
	  }
  }

  fclose(inkernel);
  
  printf("Reading convolution bias from %s\n",conv1_bias_file);
  FILE* inbias=fopen(conv1_bias_file,"r");
  k=0;
  for(unsigned i = 0; i < N_conv1_pad; i++){
	  if(i<N_conv1)
		  fscanf(inbias,"%d",&conv1_bias[0][k++]);
	  else // Padding to make it multiple of CONV_BLOCK_SIZE
		  conv1_bias[0][k++]=0;			
  }
  fclose(inbias);
 
  printf("Reading convolution weights from %s\n",conv2_wt_file);
  inkernel=fopen(conv2_wt_file,"r");
  k=0;
  for(unsigned i=0;i<N_conv2_pad;i++){
	  for(unsigned j=0;j<N_elem2;j++){
		  if(j<K_conv2*K_conv2*N_conv1/N_group2 && i<N_conv2)
			  fscanf(inkernel,"%d",&conv2_wt[0][k++]);
		  else // Padding to make it multiple of CONV_BLOCK_SIZE
			  conv2_wt[0][k++]=0;
	  }
  }
  fclose(inkernel);
  
  printf("Reading convolution bias from %s\n",conv2_bias_file);
  inbias=fopen(conv2_bias_file,"r");
  k=0;
  for(unsigned i = 0; i < N_conv2_pad; i++){
	  if(i<N_conv2)
		  fscanf(inbias,"%d",&conv2_bias[0][k++]);
	  else // Padding to make it multiple of CONV_BLOCK_SIZE
		  conv2_bias[0][k++]=0;			
  }
  fclose(inbias);
 
  printf("Reading convolution weights from %s\n",conv3_wt_file);
  inkernel=fopen(conv3_wt_file,"r");
  k=0;
  for(unsigned i=0;i<N_conv3_pad;i++){
	  for(unsigned j=0;j<N_elem3;j++){
		  if(j<K_conv3*K_conv3*N_conv2/N_group3 && i<N_conv3)
			  fscanf(inkernel,"%d",&conv3_wt[0][k++]);
		  else // Padding to make it multiple of CONV_BLOCK_SIZE
			  conv3_wt[0][k++]=0;
	  }
  }
  fclose(inkernel);
  
  printf("Reading convolution bias from %s\n",conv3_bias_file);
  inbias=fopen(conv3_bias_file,"r");
  k=0;
  for(unsigned i = 0; i < N_conv3_pad; i++){
	  if(i<N_conv3)
		  fscanf(inbias,"%d",&conv3_bias[0][k++]);
	  else // Padding to make it multiple of CONV_BLOCK_SIZE
		  conv3_bias[0][k++]=0;			
  }
  fclose(inbias);

  printf("Reading convolution weights from %s\n",conv4_wt_file);
  inkernel=fopen(conv4_wt_file,"r");
  k=0;
  for(unsigned i=0;i<N_conv4_pad;i++){
	  for(unsigned j=0;j<N_elem4;j++){
		  if(j<K_conv4*K_conv4*N_conv3/N_group4 && (i%(N_conv4_pad/N_group4)<(N_conv4/N_group4)))	// Special case for CONV_BLOCK_SIZE=128
			  fscanf(inkernel,"%d",&conv4_wt[0][k++]);
		  else // Padding to make it multiple of CONV_BLOCK_SIZE
			  conv4_wt[0][k++]=0;
	  }
  }
  fclose(inkernel);
  
  printf("Reading convolution bias from %s\n",conv4_bias_file);
  inbias=fopen(conv4_bias_file,"r");
  k=0;
  for(unsigned i = 0; i < N_conv4_pad; i++){
	  if(i%(N_conv4_pad/N_group4)<(N_conv4/N_group4))	// Special case for CONV_BLOCK_SIZE=128
		  fscanf(inbias,"%d",&conv4_bias[0][k++]);
	  else // Padding to make it multiple of CONV_BLOCK_SIZE
		  conv4_bias[0][k++]=0;			
  }
  fclose(inbias);

  printf("Reading convolution weights from %s\n",conv5_wt_file);
  inkernel=fopen(conv5_wt_file,"r");
  k=0;
  for(unsigned i=0;i<N_conv5_pad;i++){
	  for(unsigned j=0;j<N_elem5;j++){
		  if(j<K_conv5*K_conv5*N_conv4/N_group5 && i<N_conv5)
			  fscanf(inkernel,"%d",&conv5_wt[0][k++]);
		  else // Padding to make it multiple of CONV_BLOCK_SIZE
			  conv5_wt[0][k++]=0;
	  }
  }
  fclose(inkernel);
  
  printf("Reading convolution bias from %s\n",conv5_bias_file);
  inbias=fopen(conv5_bias_file,"r");
  k=0;
  for(unsigned i = 0; i < N_conv5_pad; i++){
	  if(i<N_conv5)
		  fscanf(inbias,"%d",&conv5_bias[0][k++]);
	  else // Padding to make it multiple of CONV_BLOCK_SIZE
		  conv5_bias[0][k++]=0;			
  }
  fclose(inbias);
  
  printf("Reading innerproduct weights from %s\n",fc6_wt_file);
  /*inkernel=fopen(fc6_wt_file,"r");
  for(unsigned i = 0; i < fc6_wt_dim; i++) {
	  fscanf(inkernel,"%d",&fc6_wt[0][i]);
  }*/
  inkernel=fopen(fc6_wt_file,"rb");
  fread(fc6_wt[0],sizeof(char),fc6_wt_dim,inkernel);
  fclose(inkernel);

  printf("Reading innerproduct bias from %s\n",fc6_bias_file);
  inbias=fopen(fc6_bias_file,"r");
  for(unsigned i = 0; i < fc6_out_dim; i++){
	fscanf(inbias,"%d",&fc6_bias[0][i]);
  }
  fclose(inbias);

  printf("Reading innerproduct weights from %s\n",fc7_wt_file);
  inkernel=fopen(fc7_wt_file,"rb");
  fread(fc7_wt[0],sizeof(char),fc7_wt_dim,inkernel);
  //for(unsigned i = 0; i < fc7_wt_dim; i++) 
	  //fscanf(inkernel,"%d",&fc7_wt[0][i]);  
  fclose(inkernel);
  printf("Reading innerproduct bias from %s\n",fc7_bias_file);
  inbias=fopen(fc7_bias_file,"r");
  for(unsigned i = 0; i < fc7_out_dim; i++){
	fscanf(inbias,"%d",&fc7_bias[0][i]);
  }
  fclose(inbias);
  printf("Reading innerproduct weights from %s\n",fc8_wt_file);
  inkernel=fopen(fc8_wt_file,"rb");
  fread(fc8_wt[0],sizeof(char),fc8_wt_dim,inkernel);
  //for(unsigned i = 0; i < fc8_wt_dim; i++) 
	  //fscanf(inkernel,"%d",&fc8_wt[0][i]);
  fclose(inkernel);
  printf("Reading innerproduct bias from %s\n",fc8_bias_file);
  inbias=fopen(fc8_bias_file,"r");
  for(unsigned i = 0; i < fc8_out_dim; i++){
	fscanf(inbias,"%d",&fc8_bias[0][i]);
  }
  fclose(inbias);

  printf("Reading reference labels from %s\n",inlabel_file);
  FILE* out_data=fopen(inlabel_file,"rb");
  fread(ref_label_out[0],sizeof(unsigned short),actual_test_size,out_data);
	  
  fclose(out_data);
  
  printf("Reading done\n");
}

void run() {

	cl_int status;
 
	double start_time, batch_start_time, end_time, total_time;
  
	printf("Reading Input image from %s\n",input_data);
	FILE* imagedata=fopen(input_data,"rb");
	unsigned k=0;
	fread(in_image[0],sizeof(char),actual_batch_size*no_ch*IM_W*IM_W,imagedata);
	for(unsigned img=0; img<actual_batch_size; img++)
	{
		for(unsigned i = 0; i < no_ch; i++)
		{
			for(unsigned j = 0; j < IM_W*IM_W; j++) {
				int pixel;
				pixel=(in_image[0][img*no_ch*IM_W*IM_W + i*IM_W*IM_W + j]-im_mean[i])<<QN;
				in_data[0][img*no_ch*IM_W*IM_W + i*IM_W*IM_W + j]=(short int)pixel;
			}
		}
	}
  
    // Transfer inputs to each device. Each of the host buffers supplied to
    // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
    // for the host-to-device transfer.

	status = clEnqueueWriteBuffer(queue[0], in_data_buf[0], CL_TRUE,
        0, IM_W*IM_W*no_ch*actual_batch_size * sizeof(short int), in_data[0], 0, NULL, NULL);
    checkError(status, "Failed to transfer input image");

    status = clEnqueueWriteBuffer(queue[0], conv1_wt_buf[0], CL_TRUE,
        0, N_elem1*N_conv1_pad * sizeof(char), conv1_wt[0], 0, NULL, NULL);
    checkError(status, "Failed to transfer convolution weights");
    status = clEnqueueWriteBuffer(queue[0], conv1_bias_buf[0], CL_TRUE,
        0, N_conv1_pad * sizeof(char), conv1_bias[0], 0, NULL, NULL);
    checkError(status, "Failed to transfer convolution bias");

	status = clEnqueueWriteBuffer(queue[0], conv2_wt_buf[0], CL_TRUE,
        0, N_elem2*N_conv2_pad * sizeof(char), conv2_wt[0], 0, NULL, NULL);
    checkError(status, "Failed to transfer convolution weights");
    status = clEnqueueWriteBuffer(queue[0], conv2_bias_buf[0], CL_TRUE,
        0, N_conv2_pad * sizeof(char), conv2_bias[0], 0, NULL, NULL);
    checkError(status, "Failed to transfer convolution bias");

	status = clEnqueueWriteBuffer(queue[0], conv3_wt_buf[0], CL_TRUE,
        0, N_elem3*N_conv3_pad * sizeof(char), conv3_wt[0], 0, NULL, NULL);
    checkError(status, "Failed to transfer convolution weights");
    status = clEnqueueWriteBuffer(queue[0], conv3_bias_buf[0], CL_TRUE,
        0, N_conv3_pad * sizeof(char), conv3_bias[0], 0, NULL, NULL);
    checkError(status, "Failed to transfer convolution bias");

	status = clEnqueueWriteBuffer(queue[0], conv4_wt_buf[0], CL_TRUE,
        0, N_elem4*N_conv4_pad * sizeof(char), conv4_wt[0], 0, NULL, NULL);
    checkError(status, "Failed to transfer convolution weights");
    status = clEnqueueWriteBuffer(queue[0], conv4_bias_buf[0], CL_TRUE,
        0, N_conv4_pad * sizeof(char), conv4_bias[0], 0, NULL, NULL);
    checkError(status, "Failed to transfer convolution bias");

	status = clEnqueueWriteBuffer(queue[0], conv5_wt_buf[0], CL_TRUE,
        0, N_elem5*N_conv5_pad * sizeof(char), conv5_wt[0], 0, NULL, NULL);
    checkError(status, "Failed to transfer convolution weights");
    status = clEnqueueWriteBuffer(queue[0], conv5_bias_buf[0], CL_TRUE,
        0, N_conv5_pad * sizeof(char), conv5_bias[0], 0, NULL, NULL);
    checkError(status, "Failed to transfer convolution bias");

	status = clEnqueueWriteBuffer(queue[3], fc6_wt_buf[0], CL_TRUE,
        0, fc6_wt_dim * sizeof(char), fc6_wt[0], 0, NULL, NULL);
    checkError(status, "Failed to transfer innerproduct weights");
    status = clEnqueueWriteBuffer(queue[3], fc6_bias_buf[0], CL_TRUE,
        0, fc6_out_dim * sizeof(char), fc6_bias[0], 0, NULL, NULL);
    checkError(status, "Failed to transfer innerproduct bias");

	status = clEnqueueWriteBuffer(queue[3], fc7_wt_buf[0], CL_TRUE,
        0, fc7_wt_dim * sizeof(char), fc7_wt[0], 0, NULL, NULL);
    checkError(status, "Failed to transfer innerproduct weights");
    status = clEnqueueWriteBuffer(queue[3], fc7_bias_buf[0], CL_TRUE,
        0, fc7_out_dim * sizeof(char), fc7_bias[0], 0, NULL, NULL);
    checkError(status, "Failed to transfer innerproduct bias");

	status = clEnqueueWriteBuffer(queue[3], fc8_wt_buf[0], CL_TRUE,
        0, fc8_wt_dim * sizeof(char), fc8_wt[0], 0, NULL, NULL);
    checkError(status, "Failed to transfer innerproduct weights");
    status = clEnqueueWriteBuffer(queue[3], fc8_bias_buf[0], CL_TRUE,
        0, fc8_out_dim * sizeof(char), fc8_bias[0], 0, NULL, NULL);
    checkError(status, "Failed to transfer innerproduct bias");

	// Set kernel arguments and enqueue kernel

	cl_event conv1_event[batch_size][1];
	cl_event conv2_event[batch_size][2];
	cl_event conv3_event[batch_size][1];
	cl_event conv4_event[batch_size][2];
	cl_event conv5_event[batch_size][2];
	cl_event norm_event[batch_size][2];
	cl_event pool_event[batch_size][3];
	cl_event ip_event[batch_size][3];
	cl_event label_event[batch_size][1];
	cl_event read_label_event[batch_size][1];

	unsigned argi;
    unsigned group_no=0;
	unsigned conv_wt_offset;
	unsigned conv_bias_offset;
	unsigned conv_in_offset;
	unsigned conv_out_offset;	
	unsigned no_in_group;
	unsigned no_fin_sq_pad;
	const size_t conv1_work_size[2] = {conv1_out_pad, N_conv1_pad};
	const size_t conv1_local_work_size[2] = {CONV_BLOCK_SIZE,CONV_BLOCK_SIZE};
	unsigned N_in_sq;
	const size_t norm1_work_size=1;
	const size_t pool1_work_size = N_conv1;
	const size_t conv2_work_size[2] = {conv2_out_pad, N_conv2_pad/N_group2};
	const size_t conv2_local_work_size[2] = {CONV_BLOCK_SIZE,CONV_BLOCK_SIZE};
	const size_t norm2_work_size = 1;
	const size_t pool2_work_size = N_conv2;
	const size_t conv3_work_size[2] = {conv3_out_pad, N_conv3_pad/N_group3};
	const size_t conv3_local_work_size[2] = {CONV_BLOCK_SIZE,CONV_BLOCK_SIZE};
	const size_t conv4_work_size[2] = {conv4_out_pad, N_conv4_pad/N_group4};
	const size_t conv4_local_work_size[2] = {CONV_BLOCK_SIZE,CONV_BLOCK_SIZE};
	const size_t conv5_work_size[2] = {conv5_out_pad, N_conv5_pad/N_group5};
	const size_t conv5_local_work_size[2] = {CONV_BLOCK_SIZE,CONV_BLOCK_SIZE};
	const size_t pool5_work_size = N_conv5;
	char resolution;
	char relu;
	const size_t fc6_work_size = 1;
	const size_t fc7_work_size = 1;
	const size_t fc8_work_size = 1;
	unsigned top_k;
	const size_t label_work_size = 1;
    
    start_time = getCurrentTimestamp();

	for(batch=0; batch<actual_test_size/actual_batch_size; batch++)
	{

	batch_start_time = getCurrentTimestamp();
	printf("Classifying batch-%d ",batch);
	for(image_no=0; image_no<actual_batch_size; image_no++)
	{
			
	// CONVOLUTION1

	argi=0;
	status = clSetKernelArg(kernel[0], argi++, sizeof(cl_mem), &in_data_buf[0]);
    checkError(status, "Failed to set argument %d", argi - 1);
    status = clSetKernelArg(kernel[0], argi++, sizeof(cl_mem), &conv1_wt_buf[0]);
    checkError(status, "Failed to set argument %d", argi - 1);
    status = clSetKernelArg(kernel[0], argi++, sizeof(cl_mem), &conv1_bias_buf[0]);
    checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel[0], argi++, sizeof(cl_mem), &conv1_out_buf[0]);
    checkError(status, "Failed to set argument %d", argi - 1);
	group_no=0;
	conv_wt_offset = group_no * N_conv1/N_group1 * N_elem1;
	conv_bias_offset = group_no * N_conv1/N_group1;
	conv_in_offset = (image_no*no_ch*IM_W*IM_W) + group_no * no_ch/N_group1 * IM_W*IM_W;
	conv_out_offset = group_no * N_conv1/N_group1 * conv1_out_pad;	
	status = clSetKernelArg(kernel[0], argi++, sizeof(conv_wt_offset), &conv_wt_offset);
    checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel[0], argi++, sizeof(conv_bias_offset), &conv_bias_offset);
    checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel[0], argi++, sizeof(conv_in_offset), &conv_in_offset);
    checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel[0], argi++, sizeof(conv_out_offset), &conv_out_offset);
    checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel[0], argi++, sizeof(N_elem1), &N_elem1);
    checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel[0], argi++, sizeof(K_conv1), &K_conv1);
    checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel[0], argi++, sizeof(S_conv1), &S_conv1);
    checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel[0], argi++, sizeof(P_conv1), &P_conv1);
    checkError(status, "Failed to set argument %d", argi - 1);
	no_in_group = no_ch/N_group1;
	status = clSetKernelArg(kernel[0], argi++, sizeof(no_in_group), &no_in_group);
    checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel[0], argi++, sizeof(IM_W), &IM_W);
    checkError(status, "Failed to set argument %d", argi - 1);
	no_fin_sq_pad = IM_W*IM_W;
	status = clSetKernelArg(kernel[0], argi++, sizeof(no_fin_sq_pad), &no_fin_sq_pad);
    checkError(status, "Failed to set argument %d", argi - 1);	
	status = clSetKernelArg(kernel[0], argi++, sizeof(conv1_out_dim), &conv1_out_dim);
    checkError(status, "Failed to set argument %d", argi - 1);	

	//printf("Launching kernel conv1 (%d elements x %d elements) \n", conv1_work_size[0],conv1_work_size[1]);
	if(image_no==0)
	{
		status = clEnqueueNDRangeKernel(queue[0], kernel[0], 2, NULL,
		    conv1_work_size, conv1_local_work_size, 0, NULL, &conv1_event[image_no][0]);
	}
	else
	{
		status = clEnqueueNDRangeKernel(queue[0], kernel[0], 2, NULL,
		    conv1_work_size, conv1_local_work_size, 1, &conv5_event[image_no-1][1], &conv1_event[image_no][0]);
	}
    checkError(status, "Failed to launch kernel conv1");
		
	// NORMALIZATION1

	argi=0;
	status = clSetKernelArg(kernel[1], argi++, sizeof(cl_mem), &conv1_out_buf[0]);
    checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel[1], argi++, sizeof(cl_mem), &norm1_out_buf[0]);
    checkError(status, "Failed to set argument %d", argi - 1);
	N_in_sq = conv1_out_dim*conv1_out_dim;
	status = clSetKernelArg(kernel[1], argi++, sizeof(N_in_sq), &N_in_sq);
    checkError(status, "Failed to set argument %d", argi - 1);	
	status = clSetKernelArg(kernel[1], argi++, sizeof(conv1_out_pad), &conv1_out_pad);
    checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel[1], argi++, sizeof(N_conv1), &N_conv1);
    checkError(status, "Failed to set argument %d", argi - 1);
		
    //printf("Launching kernel norm1 (%d elements)\n",norm1_work_size);	
	status = clEnqueueNDRangeKernel(queue[1], kernel[1], 1, NULL,
        &norm1_work_size, NULL, 1, &conv1_event[image_no][0], &norm_event[image_no][0]);
    checkError(status, "Failed to launch kernel norm1");	

	// POOLING1

	argi=0;
	status = clSetKernelArg(kernel[2], argi++, sizeof(cl_mem), &norm1_out_buf[0]);
    checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel[2], argi++, sizeof(cl_mem), &pool1_out_buf[0]);
    checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel[2], argi++, sizeof(conv1_out_dim), &conv1_out_dim);
    checkError(status, "Failed to set argument %d", argi - 1);
	no_fin_sq_pad = conv1_out_dim*conv1_out_dim;
	status = clSetKernelArg(kernel[2], argi++, sizeof(no_fin_sq_pad), &no_fin_sq_pad);
    checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel[2], argi++, sizeof(pool1_out_dim), &pool1_out_dim);
    checkError(status, "Failed to set argument %d", argi - 1);
		
	//printf("Launching kernel pool1 (%d elements)\n",pool1_work_size);	
	status = clEnqueueNDRangeKernel(queue[2], kernel[2], 1, NULL,
        &pool1_work_size, NULL, 1, &norm_event[image_no][0], &pool_event[image_no][0]);
    checkError(status, "Failed to launch kernel pool1");	
	
	for(group_no=0;group_no<N_group2;group_no++)
	{
		// CONVOLUTION2
		argi=0;
		status = clSetKernelArg(kernel[0], argi++, sizeof(cl_mem), &pool1_out_buf[0]);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(cl_mem), &conv2_wt_buf[0]);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(cl_mem), &conv2_bias_buf[0]);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(cl_mem), &conv2_out_buf[0]);
		checkError(status, "Failed to set argument %d", argi - 1);
		conv_wt_offset = group_no * N_conv2/N_group2 * N_elem2;
		conv_bias_offset = group_no * N_conv2/N_group2;
		conv_in_offset = group_no * N_conv1/N_group2 * pool1_out_dim*pool1_out_dim;
		conv_out_offset = group_no * N_conv2/N_group2 * conv2_out_pad;
		status = clSetKernelArg(kernel[0], argi++, sizeof(conv_wt_offset), &conv_wt_offset);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(conv_bias_offset), &conv_bias_offset);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(conv_in_offset), &conv_in_offset);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(conv_out_offset), &conv_out_offset);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(N_elem2), &N_elem2);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(K_conv2), &K_conv2);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(S_conv2), &S_conv2);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(P_conv2), &P_conv2);
		checkError(status, "Failed to set argument %d", argi - 1);
		no_in_group = N_conv1/N_group2;
		status = clSetKernelArg(kernel[0], argi++, sizeof(no_in_group), &no_in_group);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(pool1_out_dim), &pool1_out_dim);
		checkError(status, "Failed to set argument %d", argi - 1);
		no_fin_sq_pad = pool1_out_dim*pool1_out_dim;
		status = clSetKernelArg(kernel[0], argi++, sizeof(no_fin_sq_pad), &no_fin_sq_pad);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(conv2_out_dim), &conv2_out_dim);
		checkError(status, "Failed to set argument %d", argi - 1);	
	
		//printf("Launching kernel conv2 (%d elements x %d elements)\n", conv2_work_size[0],conv2_work_size[1]);
		if(group_no==0)
		{
			status = clEnqueueNDRangeKernel(queue[0], kernel[0], 2, NULL,
				conv2_work_size, conv2_local_work_size, 1, &pool_event[image_no][0], &conv2_event[image_no][group_no]);
		}
		else
		{
			status = clEnqueueNDRangeKernel(queue[0], kernel[0], 2, NULL,
				conv2_work_size, conv2_local_work_size, 1, &conv2_event[image_no][group_no-1], &conv2_event[image_no][group_no]);
		}
		checkError(status, "Failed to launch kernel conv2");
	}
	
	// NORMALIZATION2
	argi=0;
	status = clSetKernelArg(kernel[1], argi++, sizeof(cl_mem), &conv2_out_buf[0]);
    checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel[1], argi++, sizeof(cl_mem), &norm2_out_buf[0]);
    checkError(status, "Failed to set argument %d", argi - 1);
	N_in_sq = conv2_out_dim*conv2_out_dim;
	status = clSetKernelArg(kernel[1], argi++, sizeof(N_in_sq), &N_in_sq);
    checkError(status, "Failed to set argument %d", argi - 1);	
	status = clSetKernelArg(kernel[1], argi++, sizeof(conv2_out_pad), &conv2_out_pad);
    checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel[1], argi++, sizeof(N_conv2), &N_conv2);
    checkError(status, "Failed to set argument %d", argi - 1);
		
    //printf("Launching kernel norm2 (%d elements)\n",norm2_work_size);	
	status = clEnqueueNDRangeKernel(queue[1], kernel[1], 1, NULL,
        &norm2_work_size, NULL, 1, &conv2_event[image_no][1], &norm_event[image_no][1]);
    checkError(status, "Failed to launch kernel norm2");	

	// POOLING2

	argi=0;
	status = clSetKernelArg(kernel[2], argi++, sizeof(cl_mem), &norm2_out_buf[0]);
    checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel[2], argi++, sizeof(cl_mem), &pool2_out_buf[0]);
    checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel[2], argi++, sizeof(conv2_out_dim), &conv2_out_dim);
    checkError(status, "Failed to set argument %d", argi - 1);	
	no_fin_sq_pad = conv2_out_dim*conv2_out_dim;
	status = clSetKernelArg(kernel[2], argi++, sizeof(no_fin_sq_pad), &no_fin_sq_pad);
    checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel[2], argi++, sizeof(pool2_out_dim), &pool2_out_dim);
    checkError(status, "Failed to set argument %d", argi - 1);
		
	//printf("Launching kernel pool2 (%d elements)\n",pool2_work_size);	
	status = clEnqueueNDRangeKernel(queue[2], kernel[2], 1, NULL,
        &pool2_work_size, NULL, 1, &norm_event[image_no][1], &pool_event[image_no][1]);
    checkError(status, "Failed to launch kernel pool2");

	for(group_no=0;group_no<N_group3;group_no++)
	{
		// CONVOLUTION3
		argi=0;
		status = clSetKernelArg(kernel[0], argi++, sizeof(cl_mem), &pool2_out_buf[0]);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(cl_mem), &conv3_wt_buf[0]);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(cl_mem), &conv3_bias_buf[0]);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(cl_mem), &conv3_out_buf[0]);
		checkError(status, "Failed to set argument %d", argi - 1);
		conv_wt_offset = group_no * N_conv3/N_group3 * N_elem3;
		conv_bias_offset = group_no * N_conv3/N_group3;
		conv_in_offset = group_no * N_conv2/N_group3 * pool2_out_dim*pool2_out_dim;
		conv_out_offset = group_no * N_conv3/N_group3 * conv3_out_pad;	
		status = clSetKernelArg(kernel[0], argi++, sizeof(conv_wt_offset), &conv_wt_offset);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(conv_bias_offset), &conv_bias_offset);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(conv_in_offset), &conv_in_offset);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(conv_out_offset), &conv_out_offset);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(N_elem3), &N_elem3);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(K_conv3), &K_conv3);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(S_conv3), &S_conv3);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(P_conv3), &P_conv3);
		checkError(status, "Failed to set argument %d", argi - 1);
		no_in_group = N_conv2/N_group3;
		status = clSetKernelArg(kernel[0], argi++, sizeof(no_in_group), &no_in_group);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(pool2_out_dim), &pool2_out_dim);
		checkError(status, "Failed to set argument %d", argi - 1);
		no_fin_sq_pad = pool2_out_dim*pool2_out_dim;
		status = clSetKernelArg(kernel[0], argi++, sizeof(no_fin_sq_pad), &no_fin_sq_pad);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(conv3_out_dim), &conv3_out_dim);
		checkError(status, "Failed to set argument %d", argi - 1);	
		
		//printf("Launching kernel conv3 (%d elements x %d elements)\n", conv3_work_size[0],conv3_work_size[1]);
		if(group_no==0)
		{
			status = clEnqueueNDRangeKernel(queue[0], kernel[0], 2, NULL,
				conv3_work_size, conv3_local_work_size, 1, &pool_event[image_no][1], &conv3_event[image_no][group_no]);
		}
		else
		{
			status = clEnqueueNDRangeKernel(queue[0], kernel[0], 2, NULL,
				conv3_work_size, conv3_local_work_size, 1, &conv3_event[image_no][group_no-1], &conv3_event[image_no][group_no]);
		}

		checkError(status, "Failed to launch kernel conv3");
	}
	
	for(group_no=0;group_no<N_group4;group_no++)
	{
		// CONVOLUTION4
		argi=0;
		status = clSetKernelArg(kernel[0], argi++, sizeof(cl_mem), &conv3_out_buf[0]);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(cl_mem), &conv4_wt_buf[0]);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(cl_mem), &conv4_bias_buf[0]);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(cl_mem), &conv4_out_buf[0]);
		checkError(status, "Failed to set argument %d", argi - 1);
		conv_wt_offset = group_no * N_conv4/N_group4 * N_elem4;
		conv_bias_offset = group_no * N_conv4/N_group4;
		conv_in_offset = group_no * N_conv3/N_group4 * conv3_out_pad;	// special case: padding in input as it is coming from another convolution block as padded output.
		conv_out_offset = group_no * N_conv4/N_group4 * conv4_out_pad;	
		status = clSetKernelArg(kernel[0], argi++, sizeof(conv_wt_offset), &conv_wt_offset);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(conv_bias_offset), &conv_bias_offset);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(conv_in_offset), &conv_in_offset);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(conv_out_offset), &conv_out_offset);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(N_elem4), &N_elem4);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(K_conv4), &K_conv4);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(S_conv4), &S_conv4);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(P_conv4), &P_conv4);
		checkError(status, "Failed to set argument %d", argi - 1);
		no_in_group = N_conv3/N_group4;
		status = clSetKernelArg(kernel[0], argi++, sizeof(no_in_group), &no_in_group);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(conv3_out_dim), &conv3_out_dim);
		checkError(status, "Failed to set argument %d", argi - 1);
		no_fin_sq_pad = conv3_out_pad;
		status = clSetKernelArg(kernel[0], argi++, sizeof(no_fin_sq_pad), &no_fin_sq_pad);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(conv4_out_dim), &conv4_out_dim);
		checkError(status, "Failed to set argument %d", argi - 1);	
		
		//printf("Launching kernel conv4 (%d elements x %d elements)\n", conv4_work_size[0],conv4_work_size[1]);
		if(group_no==0)
		{
			status = clEnqueueNDRangeKernel(queue[0], kernel[0], 2, NULL,
				conv4_work_size, conv4_local_work_size, 1, &conv3_event[image_no][0], &conv4_event[image_no][group_no]);
		}
		else
		{
			status = clEnqueueNDRangeKernel(queue[0], kernel[0], 2, NULL,
				conv4_work_size, conv4_local_work_size, 1, &conv4_event[image_no][group_no-1], &conv4_event[image_no][group_no]);
		}

		checkError(status, "Failed to launch kernel conv4");
	}
		
	for(group_no=0;group_no<N_group5;group_no++)
	{
		// CONVOLUTION5
		argi=0;
		status = clSetKernelArg(kernel[0], argi++, sizeof(cl_mem), &conv4_out_buf[0]);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(cl_mem), &conv5_wt_buf[0]);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(cl_mem), &conv5_bias_buf[0]);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(cl_mem), &conv5_out_buf[0]);
		checkError(status, "Failed to set argument %d", argi - 1);
		conv_wt_offset = group_no * N_conv5/N_group5 * N_elem5;
		conv_bias_offset = group_no * N_conv5/N_group5;
		conv_in_offset =  group_no * N_conv4_pad/N_group5 * conv4_out_pad;	//special case for CONV_BLOCK_SIZE=128
		conv_out_offset = group_no * N_conv5/N_group5 * conv5_out_pad;	
		status = clSetKernelArg(kernel[0], argi++, sizeof(conv_wt_offset), &conv_wt_offset);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(conv_bias_offset), &conv_bias_offset);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(conv_in_offset), &conv_in_offset);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(conv_out_offset), &conv_out_offset);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(N_elem5), &N_elem5);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(K_conv5), &K_conv5);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(S_conv5), &S_conv5);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(P_conv5), &P_conv5);
		checkError(status, "Failed to set argument %d", argi - 1);
		no_in_group = N_conv4/N_group5;
		status = clSetKernelArg(kernel[0], argi++, sizeof(no_in_group), &no_in_group);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(conv4_out_dim), &conv4_out_dim);
		checkError(status, "Failed to set argument %d", argi - 1);
		no_fin_sq_pad = conv4_out_pad;
		status = clSetKernelArg(kernel[0], argi++, sizeof(no_fin_sq_pad), &no_fin_sq_pad);
		checkError(status, "Failed to set argument %d", argi - 1);
		status = clSetKernelArg(kernel[0], argi++, sizeof(conv5_out_dim), &conv5_out_dim);
		checkError(status, "Failed to set argument %d", argi - 1);	
	
		//printf("Launching kernel conv5 (%d elements x %d elements)\n", conv5_work_size[0],conv5_work_size[1]);
		if(group_no==0)
		{
			status = clEnqueueNDRangeKernel(queue[0], kernel[0], 2, NULL,
				conv5_work_size, conv5_local_work_size, 1, &conv4_event[image_no][1], &conv5_event[image_no][group_no]);
		}
		else
		{
			status = clEnqueueNDRangeKernel(queue[0], kernel[0], 2, NULL,
				conv5_work_size, conv5_local_work_size, 1, &conv5_event[image_no][group_no-1], &conv5_event[image_no][group_no]);
		}
		checkError(status, "Failed to launch kernel conv5");
	}
	
	// POOLING5

	argi=0;
	status = clSetKernelArg(kernel[2], argi++, sizeof(cl_mem), &conv5_out_buf[0]);
    checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel[2], argi++, sizeof(cl_mem), &pool5_out_buf[0]);
    checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel[2], argi++, sizeof(conv5_out_dim), &conv5_out_dim);
    checkError(status, "Failed to set argument %d", argi - 1);	
	no_fin_sq_pad = conv5_out_pad;
	status = clSetKernelArg(kernel[2], argi++, sizeof(no_fin_sq_pad), &no_fin_sq_pad);
    checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel[2], argi++, sizeof(pool5_out_dim), &pool5_out_dim);
    checkError(status, "Failed to set argument %d", argi - 1);

	//printf("Launching kernel pool5 (%d elements)\n",pool5_work_size);	
	status = clEnqueueNDRangeKernel(queue[2], kernel[2], 1, NULL,
        &pool5_work_size, NULL, 1, &conv5_event[image_no][1], &pool_event[image_no][2]);
    checkError(status, "Failed to launch kernel pool5");	

	argi = 0;
    status = clSetKernelArg(kernel[3], argi++, sizeof(cl_mem), &pool5_out_buf[0]);
    checkError(status, "Failed to set argument %d", argi - 1);
    status = clSetKernelArg(kernel[3], argi++, sizeof(cl_mem), &fc6_wt_buf[0]);
    checkError(status, "Failed to set argument %d", argi - 1);
    status = clSetKernelArg(kernel[3], argi++, sizeof(cl_mem), &fc6_bias_buf[0]);
    checkError(status, "Failed to set argument %d", argi - 1);
    status = clSetKernelArg(kernel[3], argi++, sizeof(cl_mem), &fc6_out_buf[0]);
    checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel[3], argi++, sizeof(fc6_in_dim), &fc6_in_dim);
    checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel[3], argi++, sizeof(fc6_out_dim), &fc6_out_dim);
    checkError(status, "Failed to set argument %d", argi - 1);
	resolution=1;
	status = clSetKernelArg(kernel[3], argi++, sizeof(resolution), &resolution);
    checkError(status, "Failed to set argument %d", argi - 1);
	relu=1;
	status = clSetKernelArg(kernel[3], argi++, sizeof(relu), &relu);
    checkError(status, "Failed to set argument %d", argi - 1);

	//printf("Launching kernel fc6 (%d elements)\n", fc6_work_size);
    status = clEnqueueNDRangeKernel(queue[3], kernel[3], 1, NULL,
        &fc6_work_size, NULL, 1, &pool_event[image_no][2], &ip_event[image_no][0]);
    checkError(status, "Failed to launch kernel fc6");

	argi = 0;
    status = clSetKernelArg(kernel[3], argi++, sizeof(cl_mem), &fc6_out_buf[0]);
    checkError(status, "Failed to set argument %d", argi - 1);
    status = clSetKernelArg(kernel[3], argi++, sizeof(cl_mem), &fc7_wt_buf[0]);
    checkError(status, "Failed to set argument %d", argi - 1);
    status = clSetKernelArg(kernel[3], argi++, sizeof(cl_mem), &fc7_bias_buf[0]);
    checkError(status, "Failed to set argument %d", argi - 1);
    status = clSetKernelArg(kernel[3], argi++, sizeof(cl_mem), &fc7_out_buf[0]);
    checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel[3], argi++, sizeof(fc6_out_dim), &fc6_out_dim);
    checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel[3], argi++, sizeof(fc7_out_dim), &fc7_out_dim);
    checkError(status, "Failed to set argument %d", argi - 1);
	resolution=0;
	status = clSetKernelArg(kernel[3], argi++, sizeof(resolution), &resolution);
    checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel[3], argi++, sizeof(relu), &relu);
    checkError(status, "Failed to set argument %d", argi - 1);

	//printf("Launching kernel fc7 (%d elements)\n", fc7_work_size);
    status = clEnqueueNDRangeKernel(queue[3], kernel[3], 1, NULL,
        &fc7_work_size, NULL, 1, &ip_event[image_no][0], &ip_event[image_no][1]);
    checkError(status, "Failed to launch kernel fc7");

	argi = 0;
    status = clSetKernelArg(kernel[3], argi++, sizeof(cl_mem), &fc7_out_buf[0]);
    checkError(status, "Failed to set argument %d", argi - 1);
    status = clSetKernelArg(kernel[3], argi++, sizeof(cl_mem), &fc8_wt_buf[0]);
    checkError(status, "Failed to set argument %d", argi - 1);
    status = clSetKernelArg(kernel[3], argi++, sizeof(cl_mem), &fc8_bias_buf[0]);
    checkError(status, "Failed to set argument %d", argi - 1);
    status = clSetKernelArg(kernel[3], argi++, sizeof(cl_mem), &fc8_out_buf[0]);
    checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel[3], argi++, sizeof(fc7_out_dim), &fc7_out_dim);
    checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel[3], argi++, sizeof(fc8_out_dim), &fc8_out_dim);
    checkError(status, "Failed to set argument %d", argi - 1);
	resolution=0;
	status = clSetKernelArg(kernel[3], argi++, sizeof(resolution), &resolution);
    checkError(status, "Failed to set argument %d", argi - 1);
	relu=0;
	status = clSetKernelArg(kernel[3], argi++, sizeof(relu), &relu);
    checkError(status, "Failed to set argument %d", argi - 1);

	//printf("Launching kernel fc8 (%d elements)\n", fc8_work_size);
    status = clEnqueueNDRangeKernel(queue[3], kernel[3], 1, NULL,
        &fc8_work_size, NULL, 1, &ip_event[image_no][1], &ip_event[image_no][2]);
    checkError(status, "Failed to launch kernel fc8");

	// PREDICT_LABEL : sort fc8 output and gives out TOP_K labels.

	argi = 0;
    status = clSetKernelArg(kernel[4], argi++, sizeof(cl_mem), &fc8_out_buf[0]);
    checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel[4], argi++, sizeof(cl_mem), &label_out_buf[batch*actual_batch_size + image_no]);
    checkError(status, "Failed to set argument %d", argi - 1);
	status = clSetKernelArg(kernel[4], argi++, sizeof(fc8_out_dim), &fc8_out_dim);
    checkError(status, "Failed to set argument %d", argi - 1);
	top_k = TOP_K;
	status = clSetKernelArg(kernel[4], argi++, sizeof(top_k), &top_k);
    checkError(status, "Failed to set argument %d", argi - 1);

	//printf("Launching kernel predict_label (%d elements)\n", label_work_size);
    status = clEnqueueNDRangeKernel(queue[4], kernel[4], 1, NULL,
        &label_work_size, NULL, 1, &ip_event[image_no][2], &label_event[image_no][0]);
    checkError(status, "Failed to launch kernel predict_label");

	status = clEnqueueReadBuffer(queue[4], label_out_buf[batch*actual_batch_size+image_no], CL_FALSE,
			0, TOP_K * sizeof(short int), temp_label_out[image_no], 1, &label_event[image_no][0], &read_label_event[image_no][0]);
		//clWaitForEvents(1,&read_label_event[0]);
	
	}	// End of the batch_size loop.
	
	// Read new batch of data
	
	clWaitForEvents(1,&conv1_event[actual_batch_size-1][0]);
	
	fread(in_image[0],sizeof(char),actual_batch_size*no_ch*IM_W*IM_W,imagedata);
	for(unsigned img=0; img<actual_batch_size; img++)
	{
		for(unsigned i = 0; i < no_ch; i++)
		{
			for(unsigned j = 0; j < IM_W*IM_W; j++) {
				int pixel;
				pixel=(in_image[0][img*no_ch*IM_W*IM_W + i*IM_W*IM_W + j]-im_mean[i])<<QN;
				in_data[0][img*no_ch*IM_W*IM_W + i*IM_W*IM_W + j]=(short int)pixel;
			}
		}
	}

	if(batch<(actual_test_size/actual_batch_size)-1)
	{
		status = clEnqueueWriteBuffer(queue[1], in_data_buf[0], CL_FALSE,
			0, IM_W*IM_W*no_ch*actual_batch_size * sizeof(short int), in_data[0], 1, &conv1_event[actual_batch_size-1][0], NULL);	// Transfer new batch of data to device
		checkError(status, "Failed to transfer input image");
	}

	clWaitForEvents(1,&read_label_event[actual_batch_size-1][0]);
	end_time = getCurrentTimestamp();
	batch_run_time = end_time - batch_start_time;
	printf("\tTime taken:  %0.3f ms\n", (batch_run_time) * 1e3);
	if(print_time==1)
	{
		printf("Convolution1 kernel time: %0.3f ms\n", getStartEndTime(conv1_event[actual_batch_size-1][0])*1e-6);
		printf("Convolution2 kernel time: %0.3f ms\n", (getStartEndTime(conv2_event[actual_batch_size-1][0])+getStartEndTime(conv2_event[actual_batch_size-1][1]))*1e-6);
		printf("Convolution3 kernel time: %0.3f ms\n", getStartEndTime(conv3_event[actual_batch_size-1][0])*1e-6);
		printf("Convolution4 kernel time: %0.3f ms\n", (getStartEndTime(conv4_event[actual_batch_size-1][0])+getStartEndTime(conv4_event[actual_batch_size-1][1]))*1e-6);
		printf("Convolution5 kernel time: %0.3f ms\n", (getStartEndTime(conv5_event[actual_batch_size-1][0])+getStartEndTime(conv5_event[actual_batch_size-1][1]))*1e-6);
		cl_ulong conv_time_ns = getStartEndTime(conv1_event[actual_batch_size-1][0])+getStartEndTime(conv2_event[actual_batch_size-1][0])+getStartEndTime(conv2_event[actual_batch_size-1][1])+getStartEndTime(conv3_event[actual_batch_size-1][0])+
			getStartEndTime(conv4_event[actual_batch_size-1][0])+getStartEndTime(conv4_event[actual_batch_size-1][1])+getStartEndTime(conv5_event[actual_batch_size-1][0])+getStartEndTime(conv5_event[actual_batch_size-1][1]);
		printf("Total Convolution kernel time: %0.3f ms\n\n", double(conv_time_ns) * 1e-6);

		printf("Normalization1 kernel time: %0.3f ms\n", double(getStartEndTime(norm_event[actual_batch_size-1][0])) * 1e-6);
		printf("Normalization2 kernel time: %0.3f ms\n", double(getStartEndTime(norm_event[actual_batch_size-1][1])) * 1e-6);
		cl_ulong norm_time_ns = getStartEndTime(norm_event[actual_batch_size-1][0])+getStartEndTime(norm_event[actual_batch_size-1][1]);	
		printf("Total Normalization kernel time: %0.3f ms\n\n", double(norm_time_ns) * 1e-6);

		printf("Pooling1 kernel time: %0.3f ms\n", double(getStartEndTime(pool_event[actual_batch_size-1][0])) * 1e-6);	
		printf("Pooling2 kernel time: %0.3f ms\n", double(getStartEndTime(pool_event[actual_batch_size-1][1])) * 1e-6);	
		printf("Pooling3 kernel time: %0.3f ms\n", double(getStartEndTime(pool_event[actual_batch_size-1][2])) * 1e-6);
		cl_ulong pool_time_ns = getStartEndTime(pool_event[actual_batch_size-1][0])+getStartEndTime(pool_event[actual_batch_size-1][1])+getStartEndTime(pool_event[actual_batch_size-1][2]);
		printf("Total Pooling kernel time: %0.3f ms\n\n", double(pool_time_ns) * 1e-6);

		printf("Innerproduct1 kernel time: %0.3f ms\n", double(getStartEndTime(ip_event[actual_batch_size-1][0])) * 1e-6);
		printf("Innerproduct2 kernel time: %0.3f ms\n", double(getStartEndTime(ip_event[actual_batch_size-1][1])) * 1e-6);
		printf("Innerproduct3 kernel time: %0.3f ms\n", double(getStartEndTime(ip_event[actual_batch_size-1][2])) * 1e-6);
		cl_ulong ip_time_ns = getStartEndTime(ip_event[actual_batch_size-1][0])+getStartEndTime(ip_event[actual_batch_size-1][1])+getStartEndTime(ip_event[actual_batch_size-1][2]);
		printf("Innerproduct kernel time: %0.3f ms\n\n", double(ip_time_ns) * 1e-6);

		cl_ulong label_time_ns = getStartEndTime(label_event[actual_batch_size-1][0]);
		printf("Predict_Label kernel time: %0.3f ms\n", double(label_time_ns) * 1e-6);

		cl_ulong read_label_time_ns = getStartEndTime(read_label_event[actual_batch_size-1][0]);
		printf("Label transfer time: %0.3f ms\n", double(read_label_time_ns) * 1e-6);
		cl_ulong total_time_ns = conv_time_ns + norm_time_ns + pool_time_ns + ip_time_ns + label_time_ns + read_label_time_ns;
		printf("Total kernel time: %0.3f ms\n", double(total_time_ns) * 1e-6);
	}

	for(unsigned i=0;i<actual_batch_size;i++)
	{
		clReleaseEvent(conv1_event[i][0]);
		clReleaseEvent(conv2_event[i][0]);
		clReleaseEvent(conv2_event[i][1]);
		clReleaseEvent(conv3_event[i][0]);
		clReleaseEvent(conv4_event[i][0]);
		clReleaseEvent(conv4_event[i][1]);
		clReleaseEvent(conv5_event[i][0]);
		clReleaseEvent(conv5_event[i][1]);
		clReleaseEvent(norm_event[i][0]);
		clReleaseEvent(norm_event[i][1]);
		clReleaseEvent(pool_event[i][0]);
		clReleaseEvent(pool_event[i][1]);
		clReleaseEvent(pool_event[i][2]);
		clReleaseEvent(ip_event[i][0]);
		clReleaseEvent(ip_event[i][1]);
		clReleaseEvent(ip_event[i][2]);
		clReleaseEvent(label_event[i][0]);
		clReleaseEvent(read_label_event[i][0]);
	}
	
	}	// End of batch loop

	
	end_time = getCurrentTimestamp();
	total_time = end_time - start_time;
	
	// Wall-clock time taken.
	printf("\nTime: %0.3f ms\n", (total_time) * 1e3);
	float total_ops = (N_elem1*N_conv1_pad*conv1_out_pad)+
		(N_conv1*conv1_out_dim*conv1_out_dim*(LRN+2))+
		(N_conv1*pool1_out_dim*pool1_out_dim*K_pool1*K_pool1)+
		(N_elem2*N_conv2_pad*conv2_out_pad)+
		(N_conv2*conv2_out_dim*conv2_out_dim*(LRN+2))+
		(N_conv2*pool2_out_dim*pool2_out_dim*K_pool2*K_pool2)+
 		(N_elem3*N_conv3_pad*conv3_out_pad)+
  		(N_elem4*N_conv4_pad*conv4_out_pad)+
  		(N_elem5*N_conv5_pad*conv5_out_pad)+
		(N_conv5*pool5_out_dim*pool5_out_dim*K_pool5*K_pool5)+
		fc6_wt_dim+fc7_wt_dim+fc8_wt_dim;
	
	float num_ops = (no_ch*K_conv1*K_conv1*N_conv1*conv1_out_dim*conv1_out_dim)+
		(N_conv1*conv1_out_dim*conv1_out_dim*(LRN+2))+
		(N_conv1*pool1_out_dim*pool1_out_dim*K_pool1*K_pool1)+
		(N_conv1/N_group2*K_conv2*K_conv2*N_conv2*conv2_out_dim*conv2_out_dim)+
		(N_conv2*conv2_out_dim*conv2_out_dim*(LRN+2))+
		(N_conv2*pool2_out_dim*pool2_out_dim*K_pool2*K_pool2)+
 		(N_conv2/N_group3*K_conv3*K_conv3*N_conv3*conv3_out_dim*conv3_out_dim)+
  		(N_conv3/N_group4*K_conv4*K_conv4*N_conv4*conv4_out_dim*conv4_out_dim)+
  		(N_conv4/N_group5*K_conv5*K_conv5*N_conv5*conv5_out_dim*conv5_out_dim)+
		(N_conv5*pool5_out_dim*pool5_out_dim*K_pool5*K_pool5)+
		fc6_wt_dim+fc7_wt_dim+fc8_wt_dim;
	const float ops = (float)(1.0f * num_ops*actual_test_size / total_time);
	printf("\nTotal Operations:%f\nThroughput: %.8f GOPS\nUseful Operations: %.2f%%\n\n", num_ops*actual_test_size, ops * 1e-9,100.0*num_ops/total_ops);

	for(int i=0;i<actual_test_size;i++)
	{
		status = clEnqueueReadBuffer(queue[4], label_out_buf[i], CL_FALSE,
			0, TOP_K * sizeof(short int), label_out[i], 0, NULL, &read_label_event[0][0]);
		clWaitForEvents(1,&read_label_event[0][0]);
	}

 printf("Writing label output\n");
 FILE* out_data=fopen("label.csv","w");
 for(image_no=0;image_no<actual_test_size;image_no++)
 {
	 for(unsigned i = 0; i < TOP_K; i++) {
		 fprintf(out_data,"%d,",label_out[image_no][i]);
	 }	
	 fprintf(out_data,"\n");
 }
 fclose(out_data);

 bool pass;
 int count1=0;
 int count5=0;
 for(image_no=0;image_no<actual_test_size;image_no++)
 {
	 pass=false;
	 unsigned i;
	 if(ref_label_out[0][image_no]==label_out[image_no][0])
		 count1++;
	 for(i = 0; i < TOP_K; i++) {
		 if(ref_label_out[0][image_no]==label_out[image_no][i])
		 {
			 pass = true;
			 break;
		 }
	 }
	 if(pass==true)
	 {
		 //printf("Image_no:%d detected at top-%d\n",image_no+1,i+1);
		 count5++;
	 }
	 //else
		 //printf("Image_no:%d recognition failed\n",image_no+1);
 }
 printf("\nTop-1 Accuracy: %.2f%% Top-5 Accuracy: %.2f%%\n",100.0f*count1/actual_test_size,100.0f*count5/actual_test_size);

}

// Free the resources allocated during initialization
void cleanup() {

	for(int i=0;i<num_kernels;i++){
		if(kernel && kernel[i]) {
			clReleaseKernel(kernel[i]);
		}
	    if(queue && queue[i]) {
		  clReleaseCommandQueue(queue[i]);
		}
	}

    if(in_data_buf && in_data_buf[0]) {
      clReleaseMemObject(in_data_buf[0]);
    }
    if(conv1_wt_buf && conv1_wt_buf[0]) {
      clReleaseMemObject(conv1_wt_buf[0]);
    }
    if(conv1_in_buf && conv1_in_buf[0]) {
      clReleaseMemObject(conv1_in_buf[0]);
    }

  if(program) {
    clReleaseProgram(program);
  }
  if(context) {
    clReleaseContext(context);
  }
}

