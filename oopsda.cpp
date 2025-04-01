#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <iostream>
#include <cmath>

void adjust_contrast(unsigned char* image, int width, int height, int channels, float factor) {
    int size = width * height * channels;
    for (int i = 0; i < size; i++) {
        int pixel = image[i];
        pixel = static_cast<int>((pixel - 128) * factor + 128);
        if (pixel < 0) pixel = 0;
        if (pixel > 255) pixel = 255;
        image[i] = static_cast<unsigned char>(pixel);
    }
}

int main() {
    const char* input_filename = "C:\\Users\\sree3\\Downloads\\input_image.jpg";
    const char* output_filename = "output_image.jpg";

    int width, height, channels;
    unsigned char* image = stbi_load(input_filename, &width, &height, &channels, 0);
    if (image == nullptr) {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }

    float contrast_factor = 1.0;
    adjust_contrast(image, width, height, channels, contrast_factor);

    if (!stbi_write_jpg(output_filename, width, height, channels, image, 100)) {
        std::cerr << "Error saving image!" << std::endl;
        stbi_image_free(image);
        return -1;
    }

    stbi_image_free(image);
    std::cout << "Image contrast adjusted and saved!" << std::endl;

    return 0;
}
