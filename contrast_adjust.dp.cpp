#include <opencv2/opencv.hpp>
#include <sycl/sycl.hpp>
#include <iostream>
#include <algorithm>
#include <chrono>

using namespace cv;
using namespace sycl;

void adjustContrastAndBrightness(cv::Mat& image, float contrastFactor, float brightnessFactor) {
    buffer<unsigned char, 1> image_buffer(image.data, range<1>(image.total()));
    queue q;
    q.submit([&](handler& h) {
        auto img_acc = image_buffer.get_access<access::mode::read_write>(h);
        h.parallel_for(range<1>(image.total()), [=](id<1> idx) {
            unsigned char value = img_acc[idx];
            float adjusted_value = contrastFactor * value + brightnessFactor;
            img_acc[idx] = static_cast<unsigned char>(std::min(std::max(adjusted_value, 0.0f), 255.0f));
        });
    });
    q.wait();
}

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();

    std::cout << "Attempting to load image from: C:/Users/sree3/Downloads/input_image.jpg" << std::endl;
    cv::Mat image = cv::imread("C:/Users/sree3/Downloads/input_image.jpg", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return -1;
    }

    float contrastFactor = 2.0f;
    float brightnessFactor = 50.0f;
    adjustContrastAndBrightness(image, contrastFactor, brightnessFactor);

    if (!cv::imwrite("output_image.jpg", image)) {
        std::cerr << "Error saving output image!" << std::endl;
        return -1;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> execution_time = end_time - start_time;

    std::cout << "Contrast and brightness adjustment completed and saved as output_image.jpg" << std::endl;
    std::cout << "Execution time: " << execution_time.count() << " seconds" << std::endl;

    return 0;
}
