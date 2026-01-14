#include "AddonVisionModel.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

#ifdef LLAMA_CLIP_AVAILABLE
    #include "clip.h"
    #include "ggml.h"
#endif

AddonVisionModel::AddonVisionModel(const Napi::CallbackInfo& info) : Napi::ObjectWrap<AddonVisionModel>(info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected 2 arguments: modelPath and mmprojPath").ThrowAsJavaScriptException();
        return;
    }

    if (info[0].IsString()) {
        modelPath = info[0].As<Napi::String>().Utf8Value();
    }

    if (info[1].IsString()) {
        mmprojPath = info[1].As<Napi::String>().Utf8Value();
    } else if (info[1].IsObject()) {
        Napi::Object options = info[1].As<Napi::Object>();
        if (options.Has("mmprojPath") && options.Get("mmprojPath").IsString()) {
            mmprojPath = options.Get("mmprojPath").As<Napi::String>().Utf8Value();
        }
    } else {
        Napi::TypeError::New(env, "mmprojPath must be a string").ThrowAsJavaScriptException();
        return;
    }

    if (mmprojPath.empty()) {
        Napi::TypeError::New(env, "mmprojPath must be provided").ThrowAsJavaScriptException();
        return;
    }

    detectVisionCapabilities();
}

AddonVisionModel::~AddonVisionModel() {
    dispose();
}

void AddonVisionModel::dispose() {
    if (disposed) return;

    if (clip_context != nullptr) {
#ifdef LLAMA_CLIP_AVAILABLE
        clip_free(clip_context);
#endif
        clip_context = nullptr;
    }

    visionModelLoaded = false;
    disposed = true;
}

Napi::Value AddonVisionModel::Init(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (disposed) {
        Napi::Error::New(env, "Vision model is disposed").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Promise::Deferred deferred = Napi::Promise::Deferred::New(env);

    try {
        bool visionSuccess = loadVisionModel();
        if (!visionSuccess) {
            deferred.Reject(Napi::Error::New(env, "Failed to load vision model").Value());
        } else {
            deferred.Resolve(Napi::Boolean::New(env, true));
        }
    } catch (const std::exception& e) {
        deferred.Reject(Napi::Error::New(env, e.what()).Value());
    }

    return deferred.Promise();
}

Napi::Value AddonVisionModel::Dispose(const Napi::CallbackInfo& info) {
    dispose();
    return info.Env().Undefined();
}

Napi::Value AddonVisionModel::ProcessImage(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 3) {
        Napi::TypeError::New(env, "Expected 3 arguments: imageData, width, height").ThrowAsJavaScriptException();
        return env.Null();
    }

    if (!info[0].IsTypedArray() || !info[1].IsNumber() || !info[2].IsNumber()) {
        Napi::TypeError::New(env, "Invalid argument types").ThrowAsJavaScriptException();
        return env.Null();
    }

    if (disposed) {
        Napi::Error::New(env, "Vision model is disposed").ThrowAsJavaScriptException();
        return env.Null();
    }

    if (!visionModelLoaded) {
        Napi::Error::New(env, "Vision model not loaded").ThrowAsJavaScriptException();
        return env.Null();
    }

    Napi::Uint8Array imageArray = info[0].As<Napi::Uint8Array>();
    int width = info[1].As<Napi::Number>().Int32Value();
    int height = info[2].As<Napi::Number>().Int32Value();

    try {
        std::vector<float> embedding = processImageData(imageArray.Data(), width, height, 3);

        Napi::Float32Array result = Napi::Float32Array::New(env, embedding.size());
        for (size_t i = 0; i < embedding.size(); i++) {
            result[i] = embedding[i];
        }

        return result;
    } catch (const std::exception& e) {
        Napi::Error::New(env, std::string("Image processing failed: ") + e.what()).ThrowAsJavaScriptException();
        return env.Null();
    }
}

Napi::Value AddonVisionModel::GetVisionCapabilities(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    Napi::Object caps = Napi::Object::New(env);
    caps.Set("maxImages", Napi::Number::New(env, visionCaps.maxImages));
    caps.Set("supportsImageGeneration", Napi::Boolean::New(env, visionCaps.supportsImageGeneration));

    Napi::Array formats = Napi::Array::New(env, visionCaps.supportedFormats.size());
    for (size_t i = 0; i < visionCaps.supportedFormats.size(); i++) {
        formats[i] = Napi::String::New(env, visionCaps.supportedFormats[i]);
    }
    caps.Set("supportedFormats", formats);

    Napi::Object maxRes = Napi::Object::New(env);
    maxRes.Set("width", Napi::Number::New(env, visionCaps.maxResolution.width));
    maxRes.Set("height", Napi::Number::New(env, visionCaps.maxResolution.height));
    caps.Set("maxResolution", maxRes);

    return caps;
}

bool AddonVisionModel::loadVisionModel() {
    if (visionModelLoaded) return true;

#ifdef LLAMA_CLIP_AVAILABLE
    clip_context_params params;
    params.use_gpu = false;
    params.verbosity = GGML_LOG_LEVEL_INFO;

    auto res = clip_init(mmprojPath.c_str(), params);
    if (res.ctx_v == nullptr) {
        if (res.ctx_a != nullptr) {
            clip_free(res.ctx_a);
        }
        return false;
    }

    if (res.ctx_a != nullptr && res.ctx_a != res.ctx_v) {
        clip_free(res.ctx_a);
    }

    clip_context = res.ctx_v;
    visionModelLoaded = true;
    detectVisionCapabilities();
    return true;
#else
    return false;
#endif
}

std::vector<float> AddonVisionModel::processImageData(const uint8_t* imageData, int width, int height, int channels) {
#ifdef LLAMA_CLIP_AVAILABLE
    if (!visionModelLoaded || clip_context == nullptr) {
        throw std::runtime_error("Vision model not loaded");
    }

    clip_image_u8* image = loadImageFromData(imageData, width, height, channels);
    if (!image) {
        throw std::runtime_error("Failed to load image data");
    }

    clip_image_f32_batch* processed = preprocessImage(image);
    if (!processed) {
        clip_image_u8_free(image);
        throw std::runtime_error("Failed to preprocess image");
    }

    size_t imageCount = clip_image_f32_batch_n_images(processed);
    if (imageCount < 1) {
        clip_image_u8_free(image);
        clip_image_f32_batch_free(processed);
        throw std::runtime_error("No preprocessed images available");
    }

    clip_image_f32* firstImage = clip_image_f32_get_img(processed, 0);
    if (!firstImage) {
        clip_image_u8_free(image);
        clip_image_f32_batch_free(processed);
        throw std::runtime_error("Failed to access preprocessed image");
    }

    std::vector<float> embedding = encodeImage(processed, firstImage);
    if (embedding.empty()) {
        clip_image_u8_free(image);
        clip_image_f32_batch_free(processed);
        throw std::runtime_error("Failed to encode image");
    }

    clip_image_u8_free(image);
    clip_image_f32_batch_free(processed);

    return embedding;
#else
    throw std::runtime_error("CLIP support not available - compile with LLAMA_CLIP_AVAILABLE");
#endif
}

#ifdef LLAMA_CLIP_AVAILABLE
clip_image_u8* AddonVisionModel::loadImageFromData(const uint8_t* data, int width, int height, int channels) {
    clip_image_u8* image = clip_image_u8_init();
    if (!image) return nullptr;

    const size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
    std::vector<uint8_t> rgb;
    const uint8_t* src = data;

    if (channels != 3) {
        rgb.resize(pixelCount * 3);
        for (size_t i = 0; i < pixelCount; i++) {
            const size_t srcIndex = i * static_cast<size_t>(channels);
            const uint8_t r = data[srcIndex + 0];
            const uint8_t g = channels > 1 ? data[srcIndex + 1] : r;
            const uint8_t b = channels > 2 ? data[srcIndex + 2] : r;
            const size_t dstIndex = i * 3;
            rgb[dstIndex + 0] = r;
            rgb[dstIndex + 1] = g;
            rgb[dstIndex + 2] = b;
        }
        src = rgb.data();
    }

    clip_build_img_from_pixels(src, width, height, image);
    return image;
}

clip_image_f32_batch* AddonVisionModel::preprocessImage(clip_image_u8* image) {
    if (!image) return nullptr;

    clip_image_f32_batch* batch = clip_image_f32_batch_init();
    if (!batch) {
        return nullptr;
    }

    if (!clip_image_preprocess(clip_context, image, batch)) {
        clip_image_f32_batch_free(batch);
        return nullptr;
    }

    return batch;
}

std::vector<float> AddonVisionModel::encodeImage(clip_image_f32_batch* images, clip_image_f32* referenceImage) {
    if (!images || !referenceImage || !clip_context) {
        return {};
    }

    int embed_dim = clip_n_mmproj_embd(clip_context);
    int n_tokens_out = clip_n_output_tokens(clip_context, referenceImage);
    if (embed_dim <= 0 || n_tokens_out <= 0) {
        return {};
    }

    std::vector<float> embedding(static_cast<size_t>(embed_dim) * static_cast<size_t>(n_tokens_out));

    bool success = clip_image_batch_encode(clip_context, 4, images, embedding.data());
    if (!success) {
        return {};
    }

    return embedding;
}
#endif

void AddonVisionModel::detectVisionCapabilities() {
    visionCaps.maxImages = 4;
    visionCaps.supportsImageGeneration = false;
    visionCaps.maxResolution = {1344, 1344};
    visionCaps.supportedFormats = {"image/jpeg", "image/png", "image/webp", "image/bmp"};

#ifdef LLAMA_CLIP_AVAILABLE
    if (clip_context != nullptr) {
        int size = clip_get_image_size(clip_context);
        if (size > 0) {
            visionCaps.maxResolution = {size, size};
        }
    }
#endif
}

bool AddonVisionModel::isValidImageFormat(const std::string& mimeType) {
    return std::find(visionCaps.supportedFormats.begin(), visionCaps.supportedFormats.end(), mimeType)
           != visionCaps.supportedFormats.end();
}

Napi::Function AddonVisionModel::GetClass(Napi::Env env) {
    return DefineClass(env, "AddonVisionModel", {
        InstanceMethod("init", &AddonVisionModel::Init),
        InstanceMethod("dispose", &AddonVisionModel::Dispose),
        InstanceMethod("processImage", &AddonVisionModel::ProcessImage),
        InstanceMethod("getVisionCapabilities", &AddonVisionModel::GetVisionCapabilities)
    });
}

namespace VisionUtils {
    ImageData decodeImage(const uint8_t* encodedData, size_t dataSize, const std::string& mimeType) {
        ImageData result;

#ifdef LLAMA_STBI_AVAILABLE
        int width, height, channels;
        unsigned char* data = stbi_load_from_memory(encodedData, (int)dataSize, &width, &height, &channels, 0);

        if (data) {
            size_t imageSize = (size_t)width * (size_t)height * (size_t)channels;
            result.data = std::make_unique<uint8_t[]>(imageSize);
            std::memcpy(result.data.get(), data, imageSize);
            result.width = width;
            result.height = height;
            result.channels = channels;

            stbi_image_free(data);
        }
#endif

        return result;
    }

    bool isSupportedImageFormat(const std::string& mimeType) {
        static const std::vector<std::string> supported = {
            "image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp", "image/tiff"
        };
        return std::find(supported.begin(), supported.end(), mimeType) != supported.end();
    }

    void normalizeImageData(uint8_t* data, int width, int height, int channels) {
        const float scale = 1.0f / 255.0f;
        const size_t total = static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(channels);

        for (size_t i = 0; i < total; i++) {
            data[i] = (uint8_t)(data[i] * scale * 255);
        }
    }
}
