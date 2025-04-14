__global__ void gaussian_blur_kernel(const unsigned char* input, unsigned char* output,
    int width, int height, const float* kernel, 
    int kernel_size) {
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;

if (col >= width || row >= height) return;

float sum = 0.0f;
int offset = kernel_size / 2;

for (int i = -offset; i <= offset; ++i) {
    for (int j = -offset; j <= offset; ++j) {
        int cur_row = min(max(row + i, 0), height - 1);
        int cur_col = min(max(col + j, 0), width - 1);
        float weight = kernel[(i + offset) * kernel_size + (j + offset)];
        sum += input[cur_row * width + cur_col] * weight;
    }
}

    output[row * width + col] = static_cast<unsigned char>(sum);
}
