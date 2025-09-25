#pragma once
#include "napi.h"
#include "addonGlobals.h"
#include <vector>
#include <memory>
#include <string>

// Forward declarations for Whisper/audio processing
struct whisper_context;
struct whisper_full_params;

class AddonAudioModel : public Napi::ObjectWrap<AddonAudioModel> {
    public:
        std::string audioModelPath;
        whisper_context* whisper_ctx = nullptr;
        bool audioModelLoaded = false;
        std::string currentLanguage = "auto";
        int sampleRate = 16000;

        // Audio capabilities
        struct AudioCapabilities {
            int maxAudioFiles = 1;
            std::vector<std::string> supportedFormats = {"audio/wav", "audio/mp3", "audio/flac", "audio/ogg"};
            int maxDuration = 300; // 5 minutes in seconds
            std::vector<int> supportedSampleRates = {16000, 22050, 44100, 48000};
            bool supportsSpeechToText = true;
            std::vector<std::string> supportedLanguages = {"en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"};
        } audioCaps;

        struct ProcessAudioResult {
            std::vector<float> embedding;
            std::string transcript;
            float confidence;
        };

        AddonAudioModel(const Napi::CallbackInfo& info);
        ~AddonAudioModel();
        void dispose();

        // NAPI wrapper methods
        Napi::Value Init(const Napi::CallbackInfo& info);
        Napi::Value Dispose(const Napi::CallbackInfo& info);
        Napi::Value ProcessAudio(const Napi::CallbackInfo& info);
        Napi::Value GetAudioCapabilities(const Napi::CallbackInfo& info);
        Napi::Value SetSampleRate(const Napi::CallbackInfo& info);
        Napi::Value SetLanguage(const Napi::CallbackInfo& info);

        static Napi::Function GetClass(Napi::Env env);

    private:
        bool disposed = false;

        // Internal audio processing methods
        bool loadAudioModel();
        ProcessAudioResult processAudioData(const float* audioData, size_t audioLength, bool generateTranscript = true);
        std::vector<float> extractAudioFeatures(const float* audioData, size_t audioLength);
        std::string transcribeAudio(const float* audioData, size_t audioLength);

        // Helper methods
        void detectAudioCapabilities();
        bool isValidAudioFormat(const std::string& mimeType);
        bool isValidLanguage(const std::string& language);
        std::vector<float> resampleAudio(const float* audioData, size_t length, int fromSampleRate, int toSampleRate);
        void normalizeAudio(float* audioData, size_t length);
};

// Helper functions for audio processing
namespace AudioUtils {
    struct AudioData {
        std::unique_ptr<float[]> data;
        size_t length;
        int sampleRate;
        int channels;
        float duration;
    };

    AudioData decodeAudio(const uint8_t* encodedData, size_t dataSize, const std::string& mimeType);
    bool isSupportedAudioFormat(const std::string& mimeType);
    void convertToMono(float* stereoData, size_t stereoLength, float* monoData);
    std::vector<float> applyPreEmphasis(const float* audioData, size_t length, float factor = 0.97f);
    std::vector<float> computeMelSpectrogram(const float* audioData, size_t length, int sampleRate);
}