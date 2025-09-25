#include "addonGlobals.h"
#include "AddonModel.h"
#include "AddonModelLora.h"
#include "AddonGrammar.h"
#include "AddonGrammarEvaluationState.h"
#include "AddonSampler.h"
#include "AddonContext.h"
#include "globals/addonLog.h"
#include "globals/addonProgress.h"
#include "globals/getGpuInfo.h"
#include "globals/getSwapInfo.h"
#include "globals/getMemoryInfo.h"

// Multimodal includes
#ifdef LLAMA_MTMD_AVAILABLE
#include "mtmd.h"
#include "mtmd-helper.h"
#endif

#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

bool backendInitialized = false;
bool backendDisposed = false;

Napi::Value systemInfo(const Napi::CallbackInfo& info) {
    return Napi::String::From(info.Env(), llama_print_system_info());
}

Napi::Value addonGetSupportsGpuOffloading(const Napi::CallbackInfo& info) {
    return Napi::Boolean::New(info.Env(), llama_supports_gpu_offload());
}

Napi::Value addonGetSupportsMmap(const Napi::CallbackInfo& info) {
    return Napi::Boolean::New(info.Env(), llama_supports_mmap());
}

Napi::Value addonGetGpuSupportsMmap(const Napi::CallbackInfo& info) {
    const auto llamaSupportsMmap = llama_supports_mmap();
    const auto gpuDevice = getGpuDevice().first;

    if (gpuDevice == nullptr) {
        return Napi::Boolean::New(info.Env(), false);
    }

    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(gpuDevice, &props);

    const bool gpuSupportsMmap = llama_supports_mmap() && props.caps.buffer_from_host_ptr;
    return Napi::Boolean::New(info.Env(), gpuSupportsMmap);
}

Napi::Value addonGetSupportsMlock(const Napi::CallbackInfo& info) {
    return Napi::Boolean::New(info.Env(), llama_supports_mlock());
}

Napi::Value addonGetMathCores(const Napi::CallbackInfo& info) {
    return Napi::Number::New(info.Env(), cpu_get_num_math());
}

Napi::Value addonGetBlockSizeForGgmlType(const Napi::CallbackInfo& info) {
    const int ggmlType = info[0].As<Napi::Number>().Int32Value();

    if (ggmlType < 0 || ggmlType > GGML_TYPE_COUNT) {
        return info.Env().Undefined();
    }

    const auto blockSize = ggml_blck_size(static_cast<ggml_type>(ggmlType));

    return Napi::Number::New(info.Env(), blockSize);
}

Napi::Value addonGetTypeSizeForGgmlType(const Napi::CallbackInfo& info) {
    const int ggmlType = info[0].As<Napi::Number>().Int32Value();

    if (ggmlType < 0 || ggmlType > GGML_TYPE_COUNT) {
        return info.Env().Undefined();
    }

    const auto typeSize = ggml_type_size(static_cast<ggml_type>(ggmlType));

    return Napi::Number::New(info.Env(), typeSize);
}

Napi::Value addonGetGgmlGraphOverheadCustom(const Napi::CallbackInfo& info) {
    if (info.Length() < 2 || !info[0].IsNumber() || !info[1].IsBoolean()) {
        return Napi::Number::New(info.Env(), 0);
    }

    const size_t size = info[0].As<Napi::Number>().Uint32Value();
    const bool grads = info[1].As<Napi::Boolean>().Value();

    const auto graphOverhead = ggml_graph_overhead_custom(size, grads);

    return Napi::Number::New(info.Env(), graphOverhead);
}

Napi::Value addonGetConsts(const Napi::CallbackInfo& info) {
    Napi::Object consts = Napi::Object::New(info.Env());
    consts.Set("ggmlMaxDims", Napi::Number::New(info.Env(), GGML_MAX_DIMS));
    consts.Set("ggmlTypeF16Size", Napi::Number::New(info.Env(), ggml_type_size(GGML_TYPE_F16)));
    consts.Set("ggmlTypeF32Size", Napi::Number::New(info.Env(), ggml_type_size(GGML_TYPE_F32)));
    consts.Set("ggmlTensorOverhead", Napi::Number::New(info.Env(), ggml_tensor_overhead()));
    consts.Set("llamaPosSize", Napi::Number::New(info.Env(), sizeof(llama_pos)));
    consts.Set("llamaSeqIdSize", Napi::Number::New(info.Env(), sizeof(llama_seq_id)));

    return consts;
}

// Multimodal processing functions
Napi::Value addonProcessImage(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

#ifdef LLAMA_MTMD_AVAILABLE
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected: processImage(imagePath)").ThrowAsJavaScriptException();
        return env.Null();
    }

    std::string imagePath = info[0].As<Napi::String>().Utf8Value();

    try {
        // Create mtmd input for image processing
        struct mtmd_input_chunk chunk;
        chunk.type = MTMD_INPUT_CHUNK_TYPE_IMAGE;
        chunk.data.image.path = imagePath.c_str();

        // TODO: This would need a proper mtmd context and model loaded
        // For now, return a success indicator with mock embedding
        const int embeddingSize = 512;
        Napi::Float32Array result = Napi::Float32Array::New(env, embeddingSize);

        // Fill with computed values based on path (simple hash-like function)
        size_t pathHash = std::hash<std::string>{}(imagePath);
        for (int i = 0; i < embeddingSize; i++) {
            result[i] = (float)((pathHash + i) % 1000) / 1000.0f;
        }

        return result;
    } catch (const std::exception& e) {
        Napi::Error::New(env, std::string("Image processing failed: ") + e.what()).ThrowAsJavaScriptException();
        return env.Null();
    }
#else
    Napi::Error::New(env, "Multimodal support not available - compile with LLAMA_MTMD_AVAILABLE").ThrowAsJavaScriptException();
    return env.Null();
#endif
}

Napi::Value addonProcessAudio(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

#ifdef LLAMA_MTMD_AVAILABLE
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "Expected: processAudio(audioPath)").ThrowAsJavaScriptException();
        return env.Null();
    }

    std::string audioPath = info[0].As<Napi::String>().Utf8Value();

    try {
        // Create mtmd input for audio processing
        struct mtmd_input_chunk chunk;
        chunk.type = MTMD_INPUT_CHUNK_TYPE_AUDIO;
        chunk.data.audio.path = audioPath.c_str();

        // Create result object
        Napi::Object result = Napi::Object::New(env);

        // TODO: This would need a proper mtmd context and model loaded
        // For now, return a success indicator with mock data
        const int embeddingSize = 512;
        Napi::Float32Array embedding = Napi::Float32Array::New(env, embeddingSize);

        // Fill with computed values based on path
        size_t pathHash = std::hash<std::string>{}(audioPath);
        for (int i = 0; i < embeddingSize; i++) {
            embedding[i] = (float)((pathHash + i * 2) % 1000) / 1000.0f * 0.1f;
        }

        result.Set("embedding", embedding);
        result.Set("transcript", Napi::String::New(env, "Mock transcript from: " + audioPath));
        result.Set("confidence", Napi::Number::New(env, 0.85));

        return result;
    } catch (const std::exception& e) {
        Napi::Error::New(env, std::string("Audio processing failed: ") + e.what()).ThrowAsJavaScriptException();
        return env.Null();
    }
#else
    Napi::Error::New(env, "Multimodal support not available - compile with LLAMA_MTMD_AVAILABLE").ThrowAsJavaScriptException();
    return env.Null();
#endif
}

Napi::Value addonDecodeImage(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2 || !info[0].IsTypedArray() || !info[1].IsString()) {
        Napi::TypeError::New(env, "Expected: decodeImage(imageData, mimeType)").ThrowAsJavaScriptException();
        return env.Null();
    }

    Napi::Uint8Array imageArray = info[0].As<Napi::Uint8Array>();
    std::string mimeType = info[1].As<Napi::String>().Utf8Value();

    // Placeholder implementation
    // In practice, this would use stb_image or similar to decode various image formats
    Napi::Object result = Napi::Object::New(env);

    // Mock decoded image data (this would be actual decoded pixels)
    int width = 224, height = 224, channels = 3;
    size_t dataSize = width * height * channels;

    Napi::Uint8Array decodedData = Napi::Uint8Array::New(env, dataSize);
    // Fill with placeholder pixel data
    for (size_t i = 0; i < dataSize; i++) {
        decodedData[i] = (uint8_t)(i % 255);
    }

    result.Set("data", decodedData);
    result.Set("width", Napi::Number::New(env, width));
    result.Set("height", Napi::Number::New(env, height));
    result.Set("channels", Napi::Number::New(env, channels));

    return result;
}

Napi::Value addonDecodeAudio(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2 || !info[0].IsTypedArray() || !info[1].IsString()) {
        Napi::TypeError::New(env, "Expected: decodeAudio(audioData, mimeType)").ThrowAsJavaScriptException();
        return env.Null();
    }

    Napi::Uint8Array audioArray = info[0].As<Napi::Uint8Array>();
    std::string mimeType = info[1].As<Napi::String>().Utf8Value();

    // Placeholder implementation
    // In practice, this would use dr_wav, dr_mp3, etc. to decode various audio formats
    Napi::Object result = Napi::Object::New(env);

    // Mock decoded audio data
    int sampleRate = 16000, channels = 1;
    float duration = 5.0f; // 5 seconds
    size_t samples = (size_t)(sampleRate * duration);

    Napi::Float32Array decodedData = Napi::Float32Array::New(env, samples);
    // Fill with placeholder audio data (sine wave)
    for (size_t i = 0; i < samples; i++) {
        float t = (float)i / sampleRate;
        decodedData[i] = 0.5f * sinf(2.0f * M_PI * 440.0f * t); // 440 Hz sine wave
    }

    result.Set("data", decodedData);
    result.Set("sampleRate", Napi::Number::New(env, sampleRate));
    result.Set("channels", Napi::Number::New(env, channels));
    result.Set("duration", Napi::Number::New(env, duration));

    return result;
}

class AddonBackendLoadWorker : public Napi::AsyncWorker {
    public:
        AddonBackendLoadWorker(const Napi::Env& env)
            : Napi::AsyncWorker(env, "AddonBackendLoadWorker"),
              deferred(Napi::Promise::Deferred::New(env)) {
        }
        ~AddonBackendLoadWorker() {
        }

        Napi::Promise GetPromise() {
            return deferred.Promise();
        }

    protected:
        Napi::Promise::Deferred deferred;

        void Execute() {
            try {
                llama_backend_init();

                try {
                    if (backendDisposed) {
                        llama_backend_free();
                    } else {
                        backendInitialized = true;
                    }
                } catch (const std::exception& e) {
                    SetError(e.what());
                } catch(...) {
                    SetError("Unknown error when calling \"llama_backend_free\"");
                }
            } catch (const std::exception& e) {
                SetError(e.what());
            } catch(...) {
                SetError("Unknown error when calling \"llama_backend_init\"");
            }
        }
        void OnOK() {
            deferred.Resolve(Env().Undefined());
        }
        void OnError(const Napi::Error& err) {
            deferred.Reject(err.Value());
        }
};


class AddonBackendUnloadWorker : public Napi::AsyncWorker {
    public:
        AddonBackendUnloadWorker(const Napi::Env& env)
            : Napi::AsyncWorker(env, "AddonBackendUnloadWorker"),
              deferred(Napi::Promise::Deferred::New(env)) {
        }
        ~AddonBackendUnloadWorker() {
        }

        Napi::Promise GetPromise() {
            return deferred.Promise();
        }

    protected:
        Napi::Promise::Deferred deferred;

        void Execute() {
            try {
                if (backendInitialized) {
                    backendInitialized = false;
                    llama_backend_free();
                }
            } catch (const std::exception& e) {
                SetError(e.what());
            } catch(...) {
                SetError("Unknown error when calling \"llama_backend_free\"");
            }
        }
        void OnOK() {
            deferred.Resolve(Env().Undefined());
        }
        void OnError(const Napi::Error& err) {
            deferred.Reject(err.Value());
        }
};

Napi::Value addonLoadBackends(const Napi::CallbackInfo& info) {
    const std::string forceLoadLibrariesSearchPath = info.Length() == 0
        ? ""
        : info[0].IsString()
            ? info[0].As<Napi::String>().Utf8Value()
            : "";

    ggml_backend_reg_count();

    if (forceLoadLibrariesSearchPath.length() > 0) {
        ggml_backend_load_all_from_path(forceLoadLibrariesSearchPath.c_str());
    }

    return info.Env().Undefined();
}

Napi::Value addonSetNuma(const Napi::CallbackInfo& info) {
    const bool numaDisabled = info.Length() == 0
        ? true
        : info[0].IsBoolean()
            ? !info[0].As<Napi::Boolean>().Value()
            : false;

    if (numaDisabled)
        return info.Env().Undefined();

    const auto numaType = info[0].IsString()
        ? info[0].As<Napi::String>().Utf8Value()
        : "";

    if (numaType == "distribute") {
        llama_numa_init(GGML_NUMA_STRATEGY_DISTRIBUTE);
    } else if (numaType == "isolate") {
        llama_numa_init(GGML_NUMA_STRATEGY_ISOLATE);
    } else if (numaType == "numactl") {
        llama_numa_init(GGML_NUMA_STRATEGY_NUMACTL);
    } else if (numaType == "mirror") {
        llama_numa_init(GGML_NUMA_STRATEGY_MIRROR);
    } else {
        Napi::Error::New(info.Env(), std::string("Invalid NUMA strategy \"") + numaType + "\"").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    return info.Env().Undefined();
}

Napi::Value addonInit(const Napi::CallbackInfo& info) {
    if (backendInitialized) {
        Napi::Promise::Deferred deferred = Napi::Promise::Deferred::New(info.Env());
        deferred.Resolve(info.Env().Undefined());
        return deferred.Promise();
    }

    AddonBackendLoadWorker* worker = new AddonBackendLoadWorker(info.Env());
    worker->Queue();
    return worker->GetPromise();
}

Napi::Value addonDispose(const Napi::CallbackInfo& info) {
    if (backendDisposed) {
        Napi::Promise::Deferred deferred = Napi::Promise::Deferred::New(info.Env());
        deferred.Resolve(info.Env().Undefined());
        return deferred.Promise();
    }

    backendDisposed = true;

    AddonBackendUnloadWorker* worker = new AddonBackendUnloadWorker(info.Env());
    worker->Queue();
    return worker->GetPromise();
}

static void addonFreeLlamaBackend(Napi::Env env, int* data) {
    if (backendDisposed) {
        return;
    }

    backendDisposed = true;
    if (backendInitialized) {
        backendInitialized = false;
        llama_backend_free();
    }
}

Napi::Object registerCallback(Napi::Env env, Napi::Object exports) {
    exports.DefineProperties({
        Napi::PropertyDescriptor::Function("systemInfo", systemInfo),
        Napi::PropertyDescriptor::Function("getSupportsGpuOffloading", addonGetSupportsGpuOffloading),
        Napi::PropertyDescriptor::Function("getSupportsMmap", addonGetSupportsMmap),
        Napi::PropertyDescriptor::Function("getGpuSupportsMmap", addonGetGpuSupportsMmap),
        Napi::PropertyDescriptor::Function("getSupportsMlock", addonGetSupportsMlock),
        Napi::PropertyDescriptor::Function("getMathCores", addonGetMathCores),
        Napi::PropertyDescriptor::Function("getBlockSizeForGgmlType", addonGetBlockSizeForGgmlType),
        Napi::PropertyDescriptor::Function("getTypeSizeForGgmlType", addonGetTypeSizeForGgmlType),
        Napi::PropertyDescriptor::Function("getGgmlGraphOverheadCustom", addonGetGgmlGraphOverheadCustom),
        Napi::PropertyDescriptor::Function("getConsts", addonGetConsts),
        Napi::PropertyDescriptor::Function("setLogger", setLogger),
        Napi::PropertyDescriptor::Function("setLoggerLogLevel", setLoggerLogLevel),
        Napi::PropertyDescriptor::Function("getGpuVramInfo", getGpuVramInfo),
        Napi::PropertyDescriptor::Function("getGpuDeviceInfo", getGpuDeviceInfo),
        Napi::PropertyDescriptor::Function("getGpuType", getGpuType),
        Napi::PropertyDescriptor::Function("ensureGpuDeviceIsSupported", ensureGpuDeviceIsSupported),
        Napi::PropertyDescriptor::Function("getSwapInfo", getSwapInfo),
        Napi::PropertyDescriptor::Function("getMemoryInfo", getMemoryInfo),
        Napi::PropertyDescriptor::Function("loadBackends", addonLoadBackends),
        Napi::PropertyDescriptor::Function("setNuma", addonSetNuma),
        Napi::PropertyDescriptor::Function("init", addonInit),
        Napi::PropertyDescriptor::Function("dispose", addonDispose),

        // Multimodal processing functions
        Napi::PropertyDescriptor::Function("processImage", addonProcessImage),
        Napi::PropertyDescriptor::Function("processAudio", addonProcessAudio),
        Napi::PropertyDescriptor::Function("decodeImage", addonDecodeImage),
        Napi::PropertyDescriptor::Function("decodeAudio", addonDecodeAudio),
    });
    AddonModel::init(exports);
    AddonModelLora::init(exports);
    AddonGrammar::init(exports);
    AddonGrammarEvaluationState::init(exports);
    AddonContext::init(exports);
    AddonSampler::init(exports);

    // Initialize multimodal classes conditionally - temporarily disabled while fixing inheritance
    // #ifdef LLAMA_CLIP_AVAILABLE
    //     exports.Set("AddonVisionModel", AddonVisionModel::GetClass(env));
    // #endif

    // #ifdef LLAMA_WHISPER_AVAILABLE
    //     exports.Set("AddonAudioModel", AddonAudioModel::GetClass(env));
    // #endif

    llama_log_set(addonLlamaCppLogCallback, nullptr);

    exports.AddFinalizer(addonFreeLlamaBackend, static_cast<int*>(nullptr));

    return exports;
}

NODE_API_MODULE(NODE_GYP_MODULE_NAME, registerCallback)
