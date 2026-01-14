#pragma once
#include "napi.h"
#include <memory>
#include <string>
#include <vector>

// Forward declarations for CLIP/vision processing
struct clip_ctx;
struct clip_image_u8;
struct clip_image_f32;
struct clip_image_f32_batch;

class AddonVisionModel : public Napi::ObjectWrap<AddonVisionModel> {
    public:
        std::string modelPath;
        std::string mmprojPath;
        clip_ctx* clip_context = nullptr;
        bool visionModelLoaded = false;
        bool disposed = false;

        // Vision capabilities
        struct VisionCapabilities {
            int maxImages = 1;
            std::vector<std::string> supportedFormats = {"image/jpeg", "image/png", "image/webp"};
            struct {
                int width = 1344;
                int height = 1344;
            } maxResolution;
            bool supportsImageGeneration = false;
        } visionCaps;

        AddonVisionModel(const Napi::CallbackInfo& info);
        ~AddonVisionModel();
        void dispose();

        Napi::Value Init(const Napi::CallbackInfo& info);
        Napi::Value Dispose(const Napi::CallbackInfo& info);
        Napi::Value ProcessImage(const Napi::CallbackInfo& info);
        Napi::Value GetVisionCapabilities(const Napi::CallbackInfo& info);

        static Napi::Function GetClass(Napi::Env env);

    private:
        // Internal vision processing methods
        bool loadVisionModel();
        std::vector<float> processImageData(const uint8_t* imageData, int width, int height, int channels);
        clip_image_u8* loadImageFromData(const uint8_t* data, int width, int height, int channels);
        clip_image_f32_batch* preprocessImage(clip_image_u8* image);
        std::vector<float> encodeImage(clip_image_f32_batch* images, clip_image_f32* referenceImage);

        // Helper methods
        void detectVisionCapabilities();
        bool isValidImageFormat(const std::string& mimeType);
};

// Helper functions for image processing
namespace VisionUtils {
    struct ImageData {
        std::unique_ptr<uint8_t[]> data;
        int width;
        int height;
        int channels;
    };

    ImageData decodeImage(const uint8_t* encodedData, size_t dataSize, const std::string& mimeType);
    bool isSupportedImageFormat(const std::string& mimeType);
    void normalizeImageData(uint8_t* data, int width, int height, int channels);
}
