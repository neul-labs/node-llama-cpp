import path from "path";
import {fileURLToPath} from "url";
import {
    getLlama,
    LlamaMultimodalModel,
    LlamaMultimodalChatSession,
    ImageInput,
    AudioInput
} from "node-llama-cpp";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

async function multimodalChatExample() {
    console.log("🚀 Starting Multimodal Chat Example");

    // Get the Llama instance
    const llama = await getLlama();

    try {
        // Load a multimodal model (e.g., LLaVA or Qwen2-VL)
        const model = await LlamaMultimodalModel.create({
            modelPath: path.join(__dirname, "models", "llava-v1.6-vicuna-7b.Q4_K_M.gguf"),
            mmprojPath: path.join(__dirname, "models", "llava-v1.6-vicuna-7b-mmproj.gguf"),
            enableVision: true,
            enableAudio: true,
            maxImageCache: 50,
            maxAudioCache: 25,
            contextSize: 4096
        });

        console.log("📋 Vision capabilities:", model.getVisionCapabilities());
        console.log("📋 Audio capabilities:", model.getAudioCapabilities());

        // Create a multimodal context
        const context = await model.createContext({
            contextSize: 4096,
            maxImagesInContext: 3,
            maxAudioInContext: 2
        });

        // Create a multimodal chat session
        const session = await LlamaMultimodalChatSession.create({
            context,
            systemPrompt: "You are a helpful AI assistant that can understand images, audio, and text. Describe what you see or hear, and answer questions about the content.",
            autoProcessImages: true,
            autoProcessAudio: true
        });

        // Example 1: Text + Image
        console.log("\n🖼️ Example 1: Analyzing an image");
        const imageInput: ImageInput = {
            path: path.join(__dirname, "assets", "example-image.jpg")
        };

        const response1 = await session.prompt({
            text: "What do you see in this image? Please describe it in detail.",
            images: [imageInput]
        }, {
            maxTokens: 200,
            temperature: 0.7
        });
        console.log("🤖 AI Response:", response1);

        // Example 2: Text + Audio (if audio model is available)
        const audioCaps = model.getAudioCapabilities();
        if (audioCaps.supported) {
            console.log("\n🎵 Example 2: Analyzing audio");
            const audioInput: AudioInput = {
                path: path.join(__dirname, "assets", "example-audio.wav")
            };

            const response2 = await session.prompt({
                text: "What do you hear in this audio? Please transcribe and describe it.",
                audio: [audioInput]
            }, {
                maxTokens: 200,
                temperature: 0.7
            });
            console.log("🤖 AI Response:", response2);
        }

        // Example 3: Text + Multiple Images
        console.log("\n🖼️🖼️ Example 3: Comparing multiple images");
        const imageInput1: ImageInput = {
            path: path.join(__dirname, "assets", "image1.jpg")
        };
        const imageInput2: ImageInput = {
            path: path.join(__dirname, "assets", "image2.jpg")
        };

        const response3 = await session.prompt({
            text: "Compare these two images. What are the similarities and differences?",
            images: [imageInput1, imageInput2]
        }, {
            maxTokens: 300,
            temperature: 0.8
        });
        console.log("🤖 AI Response:", response3);

        // Example 4: Demonstrate caching by reusing the same image
        console.log("\n⚡ Example 4: Reusing cached image processing");
        const response4 = await session.prompt({
            text: "Tell me something different about this same image.",
            images: [imageInput] // This will use cached embedding
        }, {
            maxTokens: 150
        });
        console.log("🤖 AI Response (using cached processing):", response4);

        // Check cache statistics
        console.log("\n📊 Cache Statistics:");
        console.log(`- Images cached: ${model.cachedImageCount}`);
        console.log(`- Audio cached: ${model.cachedAudioCount}`);

        // Clean up
        await session.dispose();
        await context.dispose();
        await model.dispose();

        console.log("\n✅ Multimodal chat example completed!");
    } catch (error) {
        console.error("❌ Error:", error);
        if (error instanceof Error) {
            console.error("Stack:", error.stack);
        }
    }
}

// Helper function to demonstrate multimodal functions
async function multimodalFunctionExample() {
    console.log("\n🔧 Function Calling Example");

    // This would be part of a larger multimodal session
    const functions = {
        analyzeImage: {
            description: "Analyze an image and extract metadata",
            params: {
                type: "object",
                properties: {
                    analysis_type: {
                        type: "string",
                        enum: ["objects", "text", "faces", "colors"]
                    }
                }
            },
            handler: (params: {analysis_type: string}, context?: {images?: any[], audio?: any[]}) => {
                console.log(`📸 Analyzing image for: ${params.analysis_type}`);
                console.log(`📸 Available images: ${context?.images?.length || 0}`);
                return {
                    result: `Performed ${params.analysis_type} analysis`,
                    confidence: 0.95
                };
            }
        }
    };

    console.log("🔧 Function defined with multimodal context support");
}

// Run the example
if (import.meta.url === `file://${process.argv[1]}`) {
    multimodalChatExample()
        .then(() => multimodalFunctionExample())
        .catch(console.error);
}
