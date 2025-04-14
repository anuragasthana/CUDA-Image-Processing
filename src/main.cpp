#include <iostream>
#include "image_processor.h"

int main(int argc, char* argv[]) {
  // Parse command line arguments
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] 
              << " <input_dir> <output_dir> [--sigma=value] [--kernel_size=value]\n";
    return 1;
  }

  // Default parameters
  float sigma = 1.0f;
  int kernel_size = 3;

  // Parse optional parameters
  for (int i = 3; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.find("--sigma=") == 0) {
      sigma = std::stof(arg.substr(8));
    } else if (arg.find("--kernel_size=") == 0) {
      kernel_size = std::stoi(arg.substr(14));
    }
  }

  // Process images
  ImageProcessor processor(argv[1], argv[2], sigma, kernel_size);
  processor.process_all_images();
  
  return 0;
}