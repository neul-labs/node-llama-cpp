#pragma once
#include "llama.h"
#include "napi.h"
#include "addonGlobals.h"
#include "AddonModel.h"
#include <vector>
#include <memory>

// Forward declarations for CLIP/vision processing
struct clip_ctx;
struct clip_image_u8;
struct clip_image_f32;

class AddonVisionModel : public AddonModel {
    public:
        std::string mmprojPath;
        clip_ctx* clip_context = nullptr;
        bool visionModelLoaded = false;

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

        // New multimodal methods
        Napi::Value ProcessImage(const Napi::CallbackInfo& info);
        Napi::Value GetVisionCapabilities(const Napi::CallbackInfo& info);

        // Override Init to load vision model
        Napi::Value Init(const Napi::CallbackInfo& info);

        static Napi::Function GetClass(Napi::Env env);

    private:
        // Internal vision processing methods
        bool loadVisionModel();
        std::vector<float> processImageData(const uint8_t* imageData, int width, int height, int channels);
        clip_image_u8* loadImageFromData(const uint8_t* data, int width, int height, int channels);
        clip_image_f32* preprocessImage(clip_image_u8* image);
        std::vector<float> encodeImage(clip_image_f32* image);

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