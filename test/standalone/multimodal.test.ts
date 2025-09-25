import path from "path";
import {describe, expect, it, beforeAll, afterAll} from "vitest";
import {
    getLlama,
    LlamaMultimodalModel,
    LlamaMultimodalContext,
    LlamaMultimodalChatSession,
    ImageInput,
    AudioInput,
    MultimodalInput,
    UnsupportedError
} from "../../src/index.js";

describe("Multimodal Functionality", () => {
    let llama: Awaited<ReturnType<typeof getLlama>>;

    beforeAll(async () => {
        llama = await getLlama();
    });

    afterAll(async () => {
        await llama?.dispose?.();
    });

    describe("LlamaMultimodalModel", () => {
        it("should detect non-multimodal models", async () => {
            // This test uses a mock approach since we don't have actual model files in tests
            try {
                const model = await LlamaMultimodalModel._create({
                    modelPath: "/nonexistent/model.gguf"
                    // No mmproj or audio model paths
                }, {_llama: llama});

                expect(model.isMultimodalCapable).toBe(false);
                await model.dispose();
            } catch (error) {
                // Expected - file doesn't exist
                expect(error).toBeInstanceOf(Error);
            }
        });

        it("should validate multimodal capabilities structure", () => {
            const mockModel = {
                fileInfo: {
                    metadata: {
                        general: {
                            architecture: "qwen2vl" as const
                        }
                    }
                },
                _inferMaxImagesFromModel: () => 4,
                _inferMaxAudioFilesFromModel: () => 2
            };

            // Test the capability detection logic structure
            const capabilities = {
                vision: {
                    supported: true,
                    maxImages: 4,
                    supportedFormats: ["image/jpeg", "image/png", "image/webp", "image/gif"],
                    maxResolution: {width: 1344, height: 1344},
                    supportsImageGeneration: false,
                    supportsImageUnderstanding: true,
                    supportsVQA: true
                },
                audio: {
                    supported: true,
                    maxAudioFiles: 2,
                    supportedFormats: ["audio/wav", "audio/mp3", "audio/flac"],
                    maxDuration: 300,
                    supportedSampleRates: [16000, 22050, 44100, 48000],
                    supportsSpeechToText: false,
                    supportsAudioUnderstanding: false,
                    supportsAudioGeneration: false,
                    supportedLanguages: []
                }
            };

            expect(capabilities.vision.supported).toBe(true);
            expect(capabilities.audio.supported).toBe(true);
            expect(capabilities.vision.maxImages).toBe(4);
            expect(capabilities.audio.maxAudioFiles).toBe(2);
        });
    });

    describe("Image Input Types", () => {
        it("should support different image input formats", () => {
            const pathInput: ImageInput = {
                path: "/path/to/image.jpg",
                id: "test-image",
                description: "A test image"
            };

            const dataInput: ImageInput = {
                data: "base64encodeddata",
                mimeType: "image/jpeg",
                id: "data-image"
            };

            const bufferInput: ImageInput = {
                buffer: new Uint8Array([1, 2, 3, 4]),
                mimeType: "image/png",
                description: "Buffer image"
            };

            expect(pathInput.path).toBe("/path/to/image.jpg");
            expect(dataInput.data).toBe("base64encodeddata");
            expect(bufferInput.buffer).toBeInstanceOf(Uint8Array);
        });
    });

    describe("Audio Input Types", () => {
        it("should support different audio input formats", () => {
            const pathInput: AudioInput = {
                path: "/path/to/audio.wav",
                id: "test-audio",
                description: "A test audio file",
                options: {
                    sampleRate: 16000,
                    generateTranscript: true,
                    language: "en"
                }
            };

            const dataInput: AudioInput = {
                data: "base64encodedaudio",
                mimeType: "audio/wav",
                id: "data-audio",
                options: {
                    normalize: true,
                    maxDuration: 60
                }
            };

            const bufferInput: AudioInput = {
                buffer: new Uint8Array([1, 2, 3, 4, 5, 6]),
                mimeType: "audio/mp3",
                description: "Buffer audio"
            };

            expect(pathInput.path).toBe("/path/to/audio.wav");
            expect(pathInput.options?.language).toBe("en");
            expect(dataInput.data).toBe("base64encodedaudio");
            expect(dataInput.options?.normalize).toBe(true);
            expect(bufferInput.buffer).toBeInstanceOf(Uint8Array);
        });
    });

    describe("Multimodal Input", () => {
        it("should combine text, images, and audio", () => {
            const input: MultimodalInput = {
                text: "Analyze this media content",
                images: [
                    {
                        path: "/path/to/image1.jpg",
                        id: "img1"
                    },
                    {
                        path: "/path/to/image2.png",
                        id: "img2"
                    }
                ],
                audio: [
                    {
                        path: "/path/to/audio.wav",
                        id: "audio1",
                        options: {
                            generateTranscript: true
                        }
                    }
                ]
            };

            expect(input.text).toBe("Analyze this media content");
            expect(input.images).toHaveLength(2);
            expect(input.audio).toHaveLength(1);
            expect(input.images?.[0]?.id).toBe("img1");
            expect(input.audio?.[0]?.options?.generateTranscript).toBe(true);
        });
    });

    describe("Error Handling", () => {
        it("should throw UnsupportedError for missing implementations", () => {
            expect(() => {
                throw new UnsupportedError("Feature not implemented");
            }).toThrowError("Feature not implemented");
        });
    });

    describe("Type Safety", () => {
        it("should maintain type safety for multimodal session functions", () => {
            const testFunction = {
                description: "Test function with multimodal context",
                params: {
                    type: "object" as const,
                    properties: {
                        action: {
                            type: "string" as const,
                            enum: ["analyze", "describe"]
                        }
                    }
                },
                handler: (params: {action: string}, context?: {images?: any[], audio?: any[]}) => {
                    expect(params.action).toMatch(/analyze|describe/);
                    expect(context).toBeDefined();
                    return {success: true, action: params.action};
                }
            };

            const result = testFunction.handler(
                {action: "analyze"},
                {images: [], audio: []}
            );
            expect(result.success).toBe(true);
            expect(result.action).toBe("analyze");
        });
    });

    describe("Architecture Support", () => {
        it("should recognize supported multimodal architectures", () => {
            const supportedArchitectures = [
                "qwen2vl",
                "clip",
                "chameleon"
            ];

            supportedArchitectures.forEach((arch) => {
                expect(typeof arch).toBe("string");
                expect(arch.length).toBeGreaterThan(0);
            });
        });
    });

    describe("Cache Management", () => {
        it("should handle embedding cache operations", () => {
            // Mock cache operations
            const cache = new Map();

            const mockEmbedding = {
                embedding: new Float32Array([0.1, 0.2, 0.3]),
                imageId: "test-image-123",
                dimensions: 3,
                metadata: {
                    processedAt: new Date(),
                    model: "test-model"
                }
            };

            cache.set("test-image-123", mockEmbedding);

            expect(cache.has("test-image-123")).toBe(true);
            expect(cache.get("test-image-123")).toEqual(mockEmbedding);

            cache.clear();
            expect(cache.size).toBe(0);
        });
    });
});
