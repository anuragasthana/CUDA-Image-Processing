#include "image_processor.h"
#include <dirent.h>
#include <sys/stat.h>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

// External declaration of CUDA kernel
extern "C" void gaussian_blur_kernel(const unsigned char* input, unsigned char* output,
                                   int width, int height, const float* kernel,
                                   int kernel_size);

ImageProcessor::ImageProcessor(const std::string& input_dir,
                               const std::string& output_dir,
                               float sigma,
                               int kernel_size)
    : input_dir_(input_dir),
      output_dir_(output_dir),
      sigma_(sigma),
      kernel_size_(kernel_size) {}

void ImageProcessor::process_all_images() {
  std::vector<std::string> image_paths = load_images(input_dir_);
  
  for (const auto& input_path : image_paths) {
    // Load image (implementation depends on your image library)
    int width, height;
    unsigned char* image_data = load_image_data(input_path, &width, &height);
    
    // Allocate memory for processed image
    unsigned char* processed_data = new unsigned char[width * height];
    
    // Apply Gaussian blur
    apply_gaussian_blur(image_data, processed_data, width, height);
    
    // Save result
    std::string output_path = output_dir_ + "/processed_" + 
                             input_path.substr(input_path.find_last_of("/\\") + 1);
    save_image(output_path, processed_data, width, height);
    
    // Cleanup
    delete[] image_data;
    delete[] processed_data;
  }
}

std::vector<std::string> ImageProcessor::load_images(const std::string& directory) {
  std::vector<std::string> image_paths;
  DIR* dir;
  struct dirent* ent;
  
  if ((dir = opendir(directory.c_str())) != nullptr) {
    while ((ent = readdir(dir)) != nullptr) {
      std::string filename = ent->d_name;
      if (filename.find(".png") != std::string::npos ||
          filename.find(".jpg") != std::string::npos) {
        image_paths.push_back(directory + "/" + filename);
      }
    }
    closedir(dir);
  }
  return image_paths;
}

void ImageProcessor::save_image(const std::string& file_path,
                               const unsigned char* data,
                               int width, int height) {
  // Implementation depends on your image library
  // Example pseudo-code:
  // stbi_write_png(file_path.c_str(), width, height, 1, data, width);
  cv::Mat image(height, width, CV_8UC1, data);
  cv::imwrite(output_path, image);
}

unsigned char* load_image_data(const std::string& file_path, int& width, int& height) {
    cv::Mat grayscaleImage = cv::imread(file_path, cv::IMREAD_GRAYSCALE);
    width = grayscaleImage.cols;
    height = grayscaleImage.rows;
    unsigned char* imageData = new unsigned char[width * height];

    memcpy(imageData, grayscaleImage.data, width * height);
    return imageData;
}

void ImageProcessor::apply_gaussian_blur(const unsigned char* input,
                                        unsigned char* output,
                                        int width, int height) {
  // Generate Gaussian kernel
  float* kernel = new float[kernel_size_ * kernel_size_];
  generate_gaussian_kernel(kernel, sigma_, kernel_size_);

  // Allocate device memory
  unsigned char *d_input, *d_output;
  float *d_kernel;
  
  cudaMalloc(&d_input, width * height * sizeof(unsigned char));
  cudaMalloc(&d_output, width * height * sizeof(unsigned char));
  cudaMalloc(&d_kernel, kernel_size_ * kernel_size_ * sizeof(float));

  // Copy data to device
  cudaMemcpy(d_input, input, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, kernel, kernel_size_ * kernel_size_ * sizeof(float), cudaMemcpyHostToDevice);

  // Configure kernel launch
  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  // Launch CUDA kernel
  gaussian_blur_kernel<<<grid, block>>>(d_input, d_output, width, height, d_kernel, kernel_size_);

  // Copy result back
  cudaMemcpy(output, d_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  // Cleanup
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_kernel);
  delete[] kernel;
}

void generate_gaussian_kernel(float* kernel, float sigma, int size) {
  float sum = 0.0f;
  int center = size / 2;
  
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      float x = i - center;
      float y = j - center;
      kernel[i * size + j] = exp(-(x*x + y*y) / (2 * sigma * sigma));
      sum += kernel[i * size + j];
    }
  }
  
  // Normalize kernel
  for (int i = 0; i < size * size; ++i) {
    kernel[i] /= sum;
  }
}
