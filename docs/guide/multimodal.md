# Multimodal Models

This guide covers how to use multimodal models in node-llama-cpp, enabling you to process images, audio, and text together with the same level of usability as other API calls.

## Overview

Multimodal support allows you to:
- Process images and audio alongside text
- Use vision models like LLaVA, Qwen2-VL, LFM2-VL, and CLIP
- Use audio models like Whisper for speech recognition
- Combine different modalities in a single conversation
- Maintain the same API patterns as text-only models

## Supported Model Architectures

- **LLaVA** - Large Language and Vision Assistant
- **Qwen2-VL** - Qwen Vision Language models
- **LFM2-VL** - Liquid AI's efficient vision-language models
- **Chameleon** - Mixed-modal models
- **CLIP** - Vision understanding models
- **Whisper** - Speech recognition models
- Any model with `.mmproj.gguf` multimodal projector support

## Supported File Formats

### Images
- JPEG, PNG, WebP, BMP, GIF
- Native resolution processing up to 512×512
- Automatic optimization for larger images

### Audio
- WAV, MP3, FLAC, OGG, M4A
- Various sample rates (16kHz, 22kHz, 44kHz, 48kHz)
- Automatic transcription when supported

## Quick Start

### Basic Setup

```typescript
import { getLlama, LlamaMultimodalModel, LlamaMultimodalChatSession } from "node-llama-cpp";

// Initialize
const llama = await getLlama();

// Create multimodal model
const model = await LlamaMultimodalModel.create({
    modelPath: "path/to/your/model.gguf",
    mmprojPath: "path/to/mmproj.gguf", // Optional multimodal projector
    enableVision: true,
    enableAudio: true,
    contextSize: 4096
});

// Create context and chat session
const context = await model.createContext();
const session = await LlamaMultimodalChatSession.create({
    context,
    systemPrompt: "You are a helpful multimodal assistant."
});
```

### Image Processing

```typescript
// Chat with a single image
const response = await session.prompt({
    text: "What do you see in this image?",
    images: [{ path: "photo.jpg" }]
});

// Multiple images
const response2 = await session.prompt({
    text: "Compare these two images",
    images: [
        { path: "image1.jpg" },
        { path: "image2.jpg" }
    ]
});

// Process image directly
const imageEmbedding = await model.processImage({
    path: "image.jpg"
});
console.log(imageEmbedding); // { data: number[], width: 224, height: 224, format: 'RGB' }
```

### Audio Processing

```typescript
// Chat with audio
const response = await session.prompt({
    text: "What was said in this recording?",
    audio: [{ path: "recording.wav" }]
});

// Process audio with transcription
const audioEmbedding = await model.processAudio({
    path: "audio.wav"
}, {
    generateTranscript: true,
    language: "en"
});
console.log(audioEmbedding.transcript); // "Hello world"
```

### Mixed Multimodal Input

```typescript
// Combine text, images, and audio
const response = await session.prompt({
    text: "Analyze both the image and audio. What's the relationship?",
    images: [{ path: "presentation_slide.jpg" }],
    audio: [{ path: "presenter_voice.wav" }]
});
```

## Advanced Usage

### Model Configuration Options

```typescript
const model = await LlamaMultimodalModel.create({
    modelPath: "model.gguf",
    mmprojPath: "mmproj.gguf",

    // Multimodal options
    enableVision: true,
    enableAudio: true,
    maxImageCache: 100,    // Cache up to 100 processed images
    maxAudioCache: 50,     // Cache up to 50 processed audio files

    // Standard model options
    contextSize: 4096,
    batchSize: 32,
    threads: 4
});
```

### Session Configuration

```typescript
const session = await LlamaMultimodalChatSession.create({
    context,
    systemPrompt: "You are a multimodal AI assistant.",

    // Generation options
    maxTokens: 1000,
    temperature: 0.7,
    topP: 0.9,

    // Multimodal options
    autoProcessImages: true,  // Automatically process images when added
    autoProcessAudio: true,   // Automatically process audio when added

    // Custom prompt templates
    promptTemplate: {
        system: "System: {content}",
        user: "User: {content}",
        assistant: "Assistant: {content}",
        imageMarker: "<image>",
        audioMarker: "<audio>"
    }
});
```

### Manual Media Processing

```typescript
// Add media to context without prompting
await session.addImage({ path: "context_image.jpg" });
await session.addAudio({ path: "background_audio.wav" });

// Then chat normally
const response = await session.prompt("What's in the context?");
```

### Input Formats

#### File Path Input
```typescript
// Simple file path
const response = await session.prompt({
    text: "Describe this",
    images: ["image.jpg"]
});
```

#### Detailed Input Objects
```typescript
const response = await session.prompt({
    text: "Process this media",
    images: [{
        path: "image.jpg",
        width: 512,
        height: 512,
        format: "RGB"
    }],
    audio: [{
        path: "audio.wav",
        sampleRate: 16000,
        duration: 30.5,
        channels: 1
    }]
});
```

#### Data Buffer Input
```typescript
import fs from "fs";

const imageData = fs.readFileSync("image.jpg");
const audioData = fs.readFileSync("audio.wav");

const response = await session.prompt({
    text: "Process this data",
    images: [{
        data: imageData,
        format: "image/jpeg"
    }],
    audio: [{
        data: audioData,
        format: "audio/wav"
    }]
});
```

## Model Capabilities

### Checking Capabilities

```typescript
// Check vision capabilities
const visionCaps = model.getVisionCapabilities();
console.log(visionCaps);
// {
//     supported: true,
//     maxImages: 4,
//     supportedFormats: ["image/jpeg", "image/png", ...],
//     maxResolution: { width: 1344, height: 1344 },
//     supportsImageUnderstanding: true,
//     supportsVQA: true
// }

// Check audio capabilities
const audioCaps = model.getAudioCapabilities();
console.log(audioCaps);
// {
//     supported: true,
//     maxDuration: 300,
//     supportedFormats: ["audio/wav", "audio/mp3", ...],
//     supportsTranscription: true
// }
```

## Performance Optimization

### Caching

The multimodal implementation includes intelligent caching:

```typescript
// Configure cache sizes
const model = await LlamaMultimodalModel.create({
    modelPath: "model.gguf",
    maxImageCache: 200,  // Increased cache for better performance
    maxAudioCache: 100
});

// Clear caches when needed
model.clearImageCache();
model.clearAudioCache();

// Check cache usage
console.log(`Images cached: ${model.cachedImageCount}`);
console.log(`Audio cached: ${model.cachedAudioCount}`);
```

### Memory Management

```typescript
// Monitor active media in context
console.log("Active images:", session.activeImages.length);
console.log("Active audio:", session.activeAudio.length);

// Clear multimodal content but keep text history
session.clearHistory(); // Clears everything including media
// OR
session.context.clearMultimodalContent(); // Clears only media, keeps text
```

## Error Handling

```typescript
try {
    const response = await session.prompt({
        text: "Analyze this",
        images: [{ path: "nonexistent.jpg" }]
    });
} catch (error) {
    if (error.message.includes("Image processing failed")) {
        console.error("Failed to process image:", error);
        // Handle gracefully - maybe try without the image
    }
}

// Check if model supports multimodal before using
if (model.getVisionCapabilities().supported) {
    // Safe to use vision features
}
```

## Best Practices

### 1. Resource Management
```typescript
// Always dispose resources
try {
    // ... multimodal operations
} finally {
    await session.dispose();
    await context.dispose();
    await model.dispose();
    await llama.dispose();
}
```

### 2. Efficient Media Usage
```typescript
// Reuse processed media across prompts
await session.addImage({ path: "reference.jpg" });

// Multiple prompts can reference the same media
const analysis1 = await session.prompt("What objects are in the image?");
const analysis2 = await session.prompt("What colors are prominent?");
const analysis3 = await session.prompt("Describe the composition");
```

### 3. Batch Processing
```typescript
// Process multiple images efficiently
const images = ["img1.jpg", "img2.jpg", "img3.jpg"];
for (const imagePath of images) {
    await session.addImage({ path: imagePath });
}

const batchAnalysis = await session.prompt("Analyze all the images I've shown you");
```

## Integration with Existing Code

Multimodal sessions are fully compatible with existing node-llama-cpp patterns:

```typescript
// Same disposal patterns
session.onDispose.createListener(() => {
    console.log("Session disposed");
});

// Same history management
const history = session.getHistory();
console.log("Conversation history:", history);

// Same generation options
const response = await session.prompt(message, {
    maxTokens: 500,
    temperature: 0.8,
    signal: abortController.signal
});
```

## Troubleshooting

### Model Loading Issues
- Ensure both model file and mmproj file paths are correct
- Verify the model supports the multimodal features you're trying to use
- Check that the model and projector files are compatible versions

### Performance Issues
- Adjust cache sizes based on your memory constraints
- Consider using quantized models for faster inference
- Monitor memory usage with multiple large images/audio files

### Format Issues
- Verify media files are in supported formats
- Check file permissions and accessibility
- Use absolute paths to avoid resolution issues

For more advanced usage and API details, see the [API Reference](../api/) documentation.