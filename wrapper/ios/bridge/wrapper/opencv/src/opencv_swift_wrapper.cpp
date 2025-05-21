#include "opencv_swift_wrapper.h"
#include <chrono>

static void* pixels;
static cv::Mat image;
static CGContextRef context = nullptr;
static CGRect rect = {};

extern "C" {
cv::Mat _getCVMatFromCGImageRef(CGImageRef cgImage) {
    size_t width = CGImageGetWidth(cgImage);
    size_t height = CGImageGetHeight(cgImage);
    // Get the number of components per pixel (RGB is 4, RGBA is 4)
    size_t bits_per_component = CGImageGetBitsPerComponent(cgImage);
    size_t bytes_per_row = CGImageGetBytesPerRow(cgImage);
    size_t bytes_per_pixel = bytes_per_row / width;

    
    // Allocate buffer to hold the pixel data
    if (!pixels) {
        pixels = malloc(height * bytes_per_row);
    }

    // Create a color space
    CGColorSpaceRef color_space = CGImageGetColorSpace(cgImage);

    // Create a context that will draw into the allocated memory
    if (context == nullptr) {
        context = CGBitmapContextCreate(
                                pixels, width, height,
                                        bits_per_component, bytes_per_row,
                                        color_space, kCGImageAlphaPremultipliedLast | kCGBitmapByteOrderDefault);

    }
    
    if (rect.size.width == 0 && rect.size.height == 0){
        rect = CGRectMake(0, 0, width, height);
    }
    // Draw the image into the context
    CGContextDrawImage(context, rect, cgImage);
    // Determine the OpenCV matrix type based on bytes per pixel
    int mat_type = (bytes_per_pixel == 4) ? CV_8UC4 : CV_8UC3;

    image = cv::Mat(height, width, mat_type, pixels);
    if (bytes_per_pixel == 4) {
        cv::cvtColor(image, image, cv::COLOR_RGBA2BGR);
    } else if (bytes_per_pixel == 3) {
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    }
    return image;
}

CGImageRef _cvMatToCGImage_wo_arg(){
    // Convert to RGBA if necessary
    if (image.channels() == 3) {
        cv::cvtColor(image, image, cv::COLOR_BGR2RGBA);
    } else if (image.channels() == 4) {
        // Do nothing
    } else {
        CV_Error(cv::Error::StsBadArg, "Image must have 3 or 4 channels");
    }

    // Create CGColorSpace
    CGColorSpaceRef color_space = CGColorSpaceCreateDeviceRGB();
    
    CGDataProviderRef provider = CGDataProviderCreateWithData(
        nullptr, image.data, image.step[0] * image.rows, nullptr);
    
    CGImageRef image_ref = CGImageCreate(
                                        image.cols,
                                        image.rows,
                                        8,
                                        8 * image.channels(),
                                        image.step[0],
                                        color_space,
                                        kCGImageAlphaNoneSkipLast |        // bitmap info
                                        kCGBitmapByteOrderDefault,
                                        provider,
                                        nullptr,
                                        false,
                                        kCGRenderingIntentDefault
    );

    CGDataProviderRelease(provider);
    CGColorSpaceRelease(color_space);

    return image_ref;
}


cv::Mat nativeConvertToBGR(const uint8_t* data, int width, int height, int formatCode) {
    return MLangeFeatureOpenCV::convertToBGR(data, width, height, formatCode);
}

}
