#include <opencv2/opencv.hpp>
#include <sycl/sycl.hpp>  // Use sycl/sycl.hpp to avoid deprecation warning
#include <iostream>
#include <algorithm>  // For std::min and std::max

using namespace cv;
using namespace sycl;

// Function to adjust contrast and brightness
void adjustContrastAndBrightness(cv::Mat& image, float contrastFactor, float brightnessFactor) {
    // Create a SYCL buffer from the image data
    buffer<unsigned char, 1> image_buffer(image.data, range<1>(image.total()));

    // Set up a SYCL queue
    queue q;

    // Submit the contrast adjustment task to the queue
    q.submit([&](handler& h) {
        // Access the image buffer for read and write
        auto img_acc = image_buffer.get_access<access::mode::read_write>(h);

        // Perform contrast and brightness adjustment
        h.parallel_for(range<1>(image.total()), [=](id<1> idx) {
            unsigned char value = img_acc[idx];
            float adjusted_value = contrastFactor * value + brightnessFactor;  // Adjust contrast and brightness
            img_acc[idx] = static_cast<unsigned char>(std::min(std::max(adjusted_value, 0.0f), 255.0f)); // Saturate value between 0 and 255
        });
    });

    // Wait for the queue to complete
    q.wait();
}

int main() {
    std::cout << "Attempting to load image from: C:/Users/sree3/Downloads/input_image.jpg" << std::endl;

    // Load the image in grayscale
    cv::Mat image = cv::imread("C:/Users/sree3/Downloads/input_image.jpg", cv::IMREAD_GRAYSCALE);  // Correct image path format
    if (image.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return -1;
    }

    // Define your contrast and brightness factors (you can adjust these values as needed)
    float contrastFactor = 2.0f;  // Adjust contrast factor (1.0 means no change)
    float brightnessFactor = 50.0f;  // Adjust brightness offset

    // Adjust contrast and brightness
    adjustContrastAndBrightness(image, contrastFactor, brightnessFactor);

    // Save the output image
    if (!cv::imwrite("output_image.jpg", image)) {
        std::cerr << "Error saving output image!" << std::endl;
        return -1;
    }

    std::cout << "Contrast and brightness adjustment completed and saved as output_image.jpg" << std::endl;

    return 0;
}
