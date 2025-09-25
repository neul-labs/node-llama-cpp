# Multimodal Examples

This page provides comprehensive examples of using multimodal models with node-llama-cpp.

## Complete Examples

### Basic Multimodal Chat

```typescript
import { getLlama, LlamaMultimodalModel, LlamaMultimodalChatSession } from "node-llama-cpp";

async function basicMultimodalChat() {
    const llama = await getLlama();

    // Create LFM2-VL model (efficient vision-language model)
    const model = await LlamaMultimodalModel.create({
        modelPath: "lfm2-vl-1.6b.q4_k_m.gguf",
        mmprojPath: "lfm2-vl-mmproj-f16.gguf",
        enableVision: true,
        contextSize: 4096
    });

    const context = await model.createContext();
    const session = await LlamaMultimodalChatSession.create({
        context,
        systemPrompt: "You are a helpful vision assistant."
    });

    // Chat with image
    const response = await session.prompt({
        text: "What's in this image?",
        images: [{ path: "./photos/vacation.jpg" }]
    });

    console.log("AI:", response);

    // Cleanup
    await session.dispose();
    await context.dispose();
    await model.dispose();
    await llama.dispose();
}
```

### Document Analysis with Vision

```typescript
import { getLlama, LlamaMultimodalModel, LlamaMultimodalChatSession } from "node-llama-cpp";

async function documentAnalysis() {
    const llama = await getLlama();

    const model = await LlamaMultimodalModel.create({
        modelPath: "llava-v1.6-34b.q4_k_m.gguf",
        mmprojPath: "llava-v1.6-34b-mmproj-f16.gguf",
        enableVision: true,
        contextSize: 8192
    });

    const context = await model.createContext();
    const session = await LlamaMultimodalChatSession.create({
        context,
        systemPrompt: "You are an expert document analyzer. Extract key information accurately."
    });

    // Analyze multiple documents
    const documents = [
        "./docs/invoice.jpg",
        "./docs/receipt.jpg",
        "./docs/contract.jpg"
    ];

    for (const doc of documents) {
        const analysis = await session.prompt({
            text: `Analyze this document. Extract key information like dates, amounts, parties involved.`,
            images: [{ path: doc }]
        });

        console.log(`Analysis of ${doc}:`);
        console.log(analysis);
        console.log("---");
    }

    await session.dispose();
    await context.dispose();
    await model.dispose();
    await llama.dispose();
}
```

### Audio Transcription and Analysis

```typescript
import { getLlama, LlamaMultimodalModel, LlamaMultimodalChatSession } from "node-llama-cpp";

async function audioAnalysis() {
    const llama = await getLlama();

    const model = await LlamaMultimodalModel.create({
        modelPath: "whisper-large-v3.gguf",
        enableAudio: true,
        contextSize: 4096
    });

    const context = await model.createContext();
    const session = await LlamaMultimodalChatSession.create({
        context,
        systemPrompt: "You are an audio analysis expert."
    });

    // Process audio with transcription
    const audioFile = "./recordings/meeting.wav";
    const analysis = await session.prompt({
        text: "Please transcribe this audio and provide a summary of key points discussed.",
        audio: [{ path: audioFile }]
    });

    console.log("Meeting Analysis:");
    console.log(analysis);

    // Follow-up questions
    const followUp = await session.prompt("What action items were mentioned?");
    console.log("Action Items:", followUp);

    await session.dispose();
    await context.dispose();
    await model.dispose();
    await llama.dispose();
}
```

### Mixed Media Analysis

```typescript
async function mixedMediaAnalysis() {
    const llama = await getLlama();

    const model = await LlamaMultimodalModel.create({
        modelPath: "qwen2-vl-7b.q4_k_m.gguf",
        mmprojPath: "qwen2-vl-7b-mmproj-f16.gguf",
        enableVision: true,
        enableAudio: true,
        contextSize: 8192
    });

    const context = await model.createContext();
    const session = await LlamaMultimodalChatSession.create({
        context,
        systemPrompt: "You are a multimedia content analyzer."
    });

    // Analyze video frame + audio combination
    const result = await session.prompt({
        text: "Analyze this video frame and audio. What's happening and what's being said?",
        images: [{ path: "./video/frame_001.jpg" }],
        audio: [{ path: "./video/audio_segment.wav" }]
    });

    console.log("Multimedia Analysis:", result);

    await session.dispose();
    await context.dispose();
    await model.dispose();
    await llama.dispose();
}
```

### Batch Image Processing

```typescript
import path from "path";
import fs from "fs/promises";

async function batchImageProcessing() {
    const llama = await getLlama();

    const model = await LlamaMultimodalModel.create({
        modelPath: "llava-phi3-mini.q4_k_m.gguf",
        mmprojPath: "llava-phi3-mini-mmproj-f16.gguf",
        enableVision: true,
        maxImageCache: 200 // Increase cache for batch processing
    });

    const context = await model.createContext();
    const session = await LlamaMultimodalChatSession.create({
        context,
        systemPrompt: "You are an image classifier and analyzer."
    });

    // Process entire directory of images
    const imageDir = "./photos";
    const imageFiles = (await fs.readdir(imageDir))
        .filter(file => /\.(jpg|jpeg|png|webp)$/i.test(file))
        .map(file => path.join(imageDir, file));

    const results = [];

    for (const imagePath of imageFiles) {
        try {
            const analysis = await session.prompt({
                text: "Classify this image and describe what you see. Be concise.",
                images: [{ path: imagePath }]
            });

            results.push({
                image: imagePath,
                analysis: analysis
            });

            console.log(`Processed: ${imagePath}`);
        } catch (error) {
            console.error(`Failed to process ${imagePath}:`, error.message);
        }
    }

    // Generate summary
    const summary = await session.prompt(
        `Based on all the images I've shown you, provide a summary of the main themes and subjects.`
    );

    console.log("Batch Processing Results:", results);
    console.log("Summary:", summary);

    await session.dispose();
    await context.dispose();
    await model.dispose();
    await llama.dispose();
}
```

### Interactive Multimodal Session

```typescript
import readline from "readline";

async function interactiveMultimodal() {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });

    const llama = await getLlama();
    const model = await LlamaMultimodalModel.create({
        modelPath: "lfm2-vl-450m.q4_k_m.gguf",
        mmprojPath: "lfm2-vl-450m-mmproj-f16.gguf",
        enableVision: true,
        enableAudio: true
    });

    const context = await model.createContext();
    const session = await LlamaMultimodalChatSession.create({
        context,
        systemPrompt: "You are a helpful multimodal assistant. Be conversational and helpful."
    });

    console.log("Multimodal Chat Started! Commands:");
    console.log("- Type text to chat");
    console.log("- Type 'image:/path/to/image' to add an image");
    console.log("- Type 'audio:/path/to/audio' to add audio");
    console.log("- Type 'quit' to exit");

    async function chat() {
        const input = await new Promise<string>((resolve) => {
            rl.question("You: ", resolve);
        });

        if (input.toLowerCase() === 'quit') {
            rl.close();
            return;
        }

        try {
            let message: any = { text: "" };

            if (input.startsWith('image:')) {
                const imagePath = input.slice(6).trim();
                message = {
                    text: "I've shared an image with you. What do you see?",
                    images: [{ path: imagePath }]
                };
            } else if (input.startsWith('audio:')) {
                const audioPath = input.slice(6).trim();
                message = {
                    text: "I've shared an audio file. What do you hear?",
                    audio: [{ path: audioPath }]
                };
            } else {
                message.text = input;
            }

            const response = await session.prompt(message);
            console.log("AI:", response);

        } catch (error) {
            console.error("Error:", error.message);
        }

        // Continue the conversation
        setImmediate(chat);
    }

    await chat();

    await session.dispose();
    await context.dispose();
    await model.dispose();
    await llama.dispose();
}
```

### Performance Monitoring Example

```typescript
async function performanceMonitoringExample() {
    const llama = await getLlama();

    const model = await LlamaMultimodalModel.create({
        modelPath: "model.gguf",
        mmprojPath: "mmproj.gguf",
        enableVision: true,
        enableAudio: true,
        maxImageCache: 50,
        maxAudioCache: 25
    });

    const context = await model.createContext();
    const session = await LlamaMultimodalChatSession.create({ context });

    // Monitor performance
    console.log("Starting performance monitoring...");

    const startTime = Date.now();

    // Process several images to show caching benefits
    const images = ["img1.jpg", "img2.jpg", "img1.jpg", "img3.jpg", "img1.jpg"];

    for (let i = 0; i < images.length; i++) {
        const iterStart = Date.now();

        await session.prompt({
            text: `Describe image ${i + 1}`,
            images: [{ path: images[i] }]
        });

        const iterEnd = Date.now();

        console.log(`Image ${i + 1} processed in ${iterEnd - iterStart}ms`);
        console.log(`Cached images: ${model.cachedImageCount}`);
        console.log(`Active images in context: ${session.activeImages.length}`);
    }

    const totalTime = Date.now() - startTime;
    console.log(`Total processing time: ${totalTime}ms`);

    // Memory usage
    console.log("Cache statistics:");
    console.log(`- Images cached: ${model.cachedImageCount}`);
    console.log(`- Audio cached: ${model.cachedAudioCount}`);

    await session.dispose();
    await context.dispose();
    await model.dispose();
    await llama.dispose();
}
```

### Error Handling Example

```typescript
async function errorHandlingExample() {
    const llama = await getLlama();

    try {
        const model = await LlamaMultimodalModel.create({
            modelPath: "model.gguf",
            mmprojPath: "mmproj.gguf",
            enableVision: true
        });

        const context = await model.createContext();
        const session = await LlamaMultimodalChatSession.create({ context });

        // Test various error scenarios
        const testCases = [
            { path: "nonexistent.jpg", description: "Non-existent file" },
            { path: "text.txt", description: "Wrong file type" },
            { data: new Uint8Array([1, 2, 3]), format: "invalid/format", description: "Invalid format" }
        ];

        for (const testCase of testCases) {
            try {
                console.log(`Testing: ${testCase.description}`);

                await session.prompt({
                    text: "What's in this image?",
                    images: [testCase]
                });

                console.log("✓ Unexpected success");
            } catch (error) {
                console.log(`✗ Expected error: ${error.message}`);

                // Handle gracefully - continue without the problematic media
                const fallback = await session.prompt("Please continue our conversation.");
                console.log("Fallback response:", fallback);
            }
        }

        await session.dispose();
        await context.dispose();
        await model.dispose();
    } catch (error) {
        console.error("Failed to initialize:", error);
    } finally {
        await llama.dispose();
    }
}
```

## Model-Specific Examples

### LFM2-VL (Liquid AI)

```typescript
// Optimized for efficiency and edge deployment
const model = await LlamaMultimodalModel.create({
    modelPath: "lfm2-vl-1.6b.q4_k_m.gguf", // or lfm2-vl-450m for ultra-efficiency
    mmprojPath: "lfm2-vl-mmproj-f16.gguf",
    enableVision: true,
    contextSize: 4096
});

// LFM2-VL excels at variable resolution processing
const response = await session.prompt({
    text: "Analyze this high-resolution image efficiently",
    images: [{ path: "high_res_photo.jpg" }] // Automatic optimization
});
```

### Qwen2-VL

```typescript
// Strong multilingual and reasoning capabilities
const model = await LlamaMultimodalModel.create({
    modelPath: "qwen2-vl-7b.q4_k_m.gguf",
    mmprojPath: "qwen2-vl-7b-mmproj-f16.gguf",
    enableVision: true,
    contextSize: 8192 // Larger context for complex reasoning
});

const response = await session.prompt({
    text: "请分析这张图片中的文字内容。", // Chinese prompt
    images: [{ path: "chinese_document.jpg" }]
});
```

### LLaVA

```typescript
// Excellent for general vision-language tasks
const model = await LlamaMultimodalModel.create({
    modelPath: "llava-v1.6-34b.q4_k_m.gguf",
    mmprojPath: "llava-v1.6-34b-mmproj-f16.gguf",
    enableVision: true,
    contextSize: 8192
});

const response = await session.prompt({
    text: "Provide a detailed analysis of this scientific diagram",
    images: [{ path: "scientific_chart.png" }]
});
```

## Running the Examples

To run these examples:

1. Install dependencies:
```bash
npm install node-llama-cpp
```

2. Download appropriate models:
```bash
# Example: Download LFM2-VL
wget https://huggingface.co/LiquidAI/LFM2-VL-1.6B-GGUF/resolve/main/lfm2-vl-1.6b.q4_k_m.gguf
wget https://huggingface.co/LiquidAI/LFM2-VL-1.6B-GGUF/resolve/main/lfm2-vl-mmproj-f16.gguf
```

3. Update paths in the examples to point to your downloaded models

4. Run with:
```bash
node your-example.js
```

For more details, see the main [Multimodal Guide](./multimodal.md).