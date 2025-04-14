#ifndef IMAGE_PROCESSOR_H_
#define IMAGE_PROCESSOR_H_

#include <string>
#include <vector>

// ImageProcessor class declaration
class ImageProcessor {
 public:
  // Constructor
  ImageProcessor(const std::string& input_dir, const std::string& output_dir, float sigma, int kernel_size);

  // Method to process all images in the input directory
  void process_all_images();

 private:
  // Helper method to load images from the input directory
  std::vector<std::string> loa2d_images(const std::string& directory);

  // Helper method to save processed images to the output directory
  void save_image(const std::string& file_path, const unsigned char* data, int width, int height);

  // CUDA method to apply Gaussian blur
  void apply_gaussian_blur(const unsigned char* input, unsigned char* output, int width, int height);

  // Member variables
  std::string input_dir_;
  std::string output_dir_;
  float sigma_;
  int kernel_size_;
};

#endif  // IMAGE_PROCESSOR_H_
