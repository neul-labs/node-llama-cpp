import {DisposedError, EventRelay} from "lifecycle-utils";
import {LlamaModel} from "../LlamaModel/LlamaModel.js";
import type {LlamaModelOptions} from "../LlamaModel/LlamaModel.js";
import {ImageInput, AudioInput, ImageEmbedding, AudioEmbedding, VisionModelCapabilities, AudioModelCapabilities} from "../../types/MultimodalTypes.js";
import {LlamaText} from "../../utils/LlamaText.js";
import {Llama} from "../../bindings/Llama.js";

export type LlamaMultimodalModelOptions = LlamaModelOptions & {
    /** Path to the multimodal projector file (mmproj.gguf) */
    mmprojPath?: string;

    /** Enable vision processing */
    enableVision?: boolean;

    /** Enable audio processing */
    enableAudio?: boolean;

    /** Maximum number of images to cache */
    maxImageCache?: number;

    /** Maximum number of audio files to cache */
    maxAudioCache?: number;
};

/**
 * LlamaMultimodalModel extends LlamaModel with multimodal capabilities for processing images and audio
 * alongside text. It provides the same level of usability as other API calls in the library.
 */
export class LlamaMultimodalModel {
    /** @internal */ private readonly _baseModel: LlamaModel;
    /** @internal */ private readonly _llama: Llama;
    /** @internal */ private readonly _mmprojPath?: string;
    /** @internal */ private readonly _enableVision: boolean;
    /** @internal */ private readonly _enableAudio: boolean;
    /** @internal */ private readonly _imageCache = new Map<string, ImageEmbedding>();
    /** @internal */ private readonly _audioCache = new Map<string, AudioEmbedding>();
    /** @internal */ private readonly _maxImageCache: number;
    /** @internal */ private readonly _maxAudioCache: number;
    /** @internal */ private _disposed = false;

    public readonly onDispose = new EventRelay<void>();

    private constructor(baseModel: LlamaModel, options: LlamaMultimodalModelOptions) {
        this._baseModel = baseModel;
        this._llama = baseModel.llama;
        this._mmprojPath = options.mmprojPath;
        this._enableVision = options.enableVision ?? true;
        this._enableAudio = options.enableAudio ?? true;
        this._maxImageCache = options.maxImageCache ?? 100;
        this._maxAudioCache = options.maxAudioCache ?? 50;

        // Listen for base model disposal
        this._baseModel.onDispose.createListener(() => {
            void this.dispose();
        });
    }

    /**
     * Create a new LlamaMultimodalModel instance
     */
    public static async create(options: LlamaMultimodalModelOptions): Promise<LlamaMultimodalModel> {
        // Create the base text model first
        const baseModel = await LlamaModel.create(options);

        // Create the multimodal wrapper
        const multimodalModel = new LlamaMultimodalModel(baseModel, options);

        // Initialize multimodal components
        await multimodalModel._initializeMultimodal();

        return multimodalModel;
    }

    /**
     * Process an image and return its embedding
     */
    public async processImage(input: ImageInput): Promise<ImageEmbedding> {
        this._ensureNotDisposed();

        if (!this._enableVision) {
            throw new Error("Vision processing is not enabled for this model");
        }

        const cacheKey = this._getImageCacheKey(input);

        // Check cache first
        const cached = this._imageCache.get(cacheKey);
        if (cached) {
            return cached;
        }

        // Process the image
        const embedding = await this._processImageInternal(input);

        // Cache the result
        this._cacheImageEmbedding(cacheKey, embedding);

        return embedding;
    }

    /**
     * Process audio and return its embedding and optionally transcript
     */
    public async processAudio(input: AudioInput, options?: {
        generateTranscript?: boolean;
        language?: string;
    }): Promise<AudioEmbedding> {
        this._ensureNotDisposed();

        if (!this._enableAudio) {
            throw new Error("Audio processing is not enabled for this model");
        }

        const cacheKey = this._getAudioCacheKey(input, options);

        // Check cache first
        const cached = this._audioCache.get(cacheKey);
        if (cached) {
            return cached;
        }

        // Process the audio
        const embedding = await this._processAudioInternal(input, options);

        // Cache the result
        this._cacheAudioEmbedding(cacheKey, embedding);

        return embedding;
    }

    /**
     * Get vision model capabilities
     */
    public getVisionCapabilities(): VisionModelCapabilities {
        this._ensureNotDisposed();

        return {
            supported: this._enableVision,
            maxImages: 4,
            supportedFormats: ["image/jpeg", "image/png", "image/webp", "image/bmp"],
            maxResolution: { width: 1344, height: 1344 },
            supportsImageGeneration: false,
            supportsImageUnderstanding: true,
            supportsVQA: true
        };
    }

    /**
     * Get audio model capabilities
     */
    public getAudioCapabilities(): AudioModelCapabilities {
        this._ensureNotDisposed();

        return {
            supported: this._enableAudio,
            maxDuration: 300, // 5 minutes
            supportedFormats: ["audio/wav", "audio/mp3", "audio/flac", "audio/ogg"],
            sampleRates: [16000, 22050, 44100, 48000],
            supportsTranscription: true,
            supportsAudioGeneration: false,
            supportsVoiceCloning: false
        };
    }

    /**
     * Clear the image cache
     */
    public clearImageCache(): void {
        this._ensureNotDisposed();
        this._imageCache.clear();
    }

    /**
     * Clear the audio cache
     */
    public clearAudioCache(): void {
        this._ensureNotDisposed();
        this._audioCache.clear();
    }

    /**
     * Get cached image count
     */
    public get cachedImageCount(): number {
        return this._imageCache.size;
    }

    /**
     * Get cached audio count
     */
    public get cachedAudioCount(): number {
        return this._audioCache.size;
    }

    // Delegate all base model properties and methods
    public get contextSize(): number { return this._baseModel.contextSize; }
    public get embeddingVectorSize(): number { return this._baseModel.embeddingVectorSize; }
    public get trainContextSize(): number { return this._baseModel.trainContextSize; }
    public get fileInfo() { return this._baseModel.fileInfo; }
    public get tokenizer() { return this._baseModel.tokenizer; }
    public get vocabularyType() { return this._baseModel.vocabularyType; }
    public get isDisposed() { return this._disposed || this._baseModel.isDisposed; }

    /**
     * Create a context for text and multimodal generation
     */
    public async createContext(options?: any) {
        this._ensureNotDisposed();
        // Import the LlamaMultimodalContext dynamically to avoid circular dependency
        const { LlamaMultimodalContext } = await import("../LlamaMultimodalContext/LlamaMultimodalContext.js");
        return await LlamaMultimodalContext.create(this, options);
    }

    /**
     * Create an embedding context for multimodal embeddings
     */
    public async createEmbeddingContext(options?: any) {
        this._ensureNotDisposed();
        return await this._baseModel.createEmbeddingContext(options);
    }

    /**
     * Dispose of the multimodal model and free resources
     */
    public async dispose(): Promise<void> {
        if (this._disposed) {
            return;
        }

        this._disposed = true;
        this._imageCache.clear();
        this._audioCache.clear();

        await this._baseModel.dispose();
        this.onDispose.dispatchEvent();
    }

    /** @internal */
    private _ensureNotDisposed(): void {
        if (this._disposed) {
            throw new DisposedError();
        }
    }

    /** @internal */
    private async _initializeMultimodal(): Promise<void> {
        // TODO: Initialize multimodal projector if path is provided
        // This would load the mmproj.gguf file and set up vision/audio processing
        if (this._mmprojPath) {
            // await this._loadMultimodalProjector(this._mmprojPath);
        }
    }

    /** @internal */
    private async _processImageInternal(input: ImageInput): Promise<ImageEmbedding> {
        try {
            let imagePath: string;

            if (typeof input === 'string') {
                imagePath = input;
            } else if (input.path) {
                imagePath = input.path;
            } else {
                // Handle base64 or buffer data by saving to temp file
                imagePath = await this._saveImageToTemp(input);
            }

            // Call native processImage function
            const bindings = this._llama._bindings;
            const embedding = bindings.processImage(imagePath);

            return {
                data: Array.from(embedding),
                width: input.width || 224,
                height: input.height || 224,
                format: input.format || 'RGB',
                processingTime: Date.now() // Mock timestamp
            };
        } catch (error) {
            throw new Error(`Image processing failed: ${error instanceof Error ? error.message : String(error)}`);
        }
    }

    /** @internal */
    private async _processAudioInternal(input: AudioInput, options?: {
        generateTranscript?: boolean;
        language?: string;
    }): Promise<AudioEmbedding> {
        try {
            let audioPath: string;

            if (typeof input === 'string') {
                audioPath = input;
            } else if (input.path) {
                audioPath = input.path;
            } else {
                // Handle buffer data by saving to temp file
                audioPath = await this._saveAudioToTemp(input);
            }

            // Call native processAudio function
            const bindings = this._llama._bindings;
            const result = bindings.processAudio(audioPath);

            return {
                data: Array.from(result.embedding),
                sampleRate: input.sampleRate || 16000,
                duration: input.duration || 0,
                channels: input.channels || 1,
                transcript: result.transcript,
                confidence: result.confidence,
                processingTime: Date.now()
            };
        } catch (error) {
            throw new Error(`Audio processing failed: ${error instanceof Error ? error.message : String(error)}`);
        }
    }

    /** @internal */
    private _getImageCacheKey(input: ImageInput): string {
        if (typeof input === 'string') {
            return input;
        }

        if (input.path) {
            return input.path;
        }

        // For data inputs, create hash-based key
        const dataStr = input.data?.toString() || '';
        return `data:${input.format}:${dataStr.substring(0, 100)}`;
    }

    /** @internal */
    private _getAudioCacheKey(input: AudioInput, options?: any): string {
        let baseKey: string;

        if (typeof input === 'string') {
            baseKey = input;
        } else if (input.path) {
            baseKey = input.path;
        } else {
            const dataStr = input.data?.toString() || '';
            baseKey = `data:${input.format}:${dataStr.substring(0, 100)}`;
        }

        const optionsStr = JSON.stringify(options || {});
        return `${baseKey}:${optionsStr}`;
    }

    /** @internal */
    private _cacheImageEmbedding(key: string, embedding: ImageEmbedding): void {
        if (this._imageCache.size >= this._maxImageCache) {
            // Remove oldest entry (simple FIFO)
            const firstKey = this._imageCache.keys().next().value;
            this._imageCache.delete(firstKey);
        }
        this._imageCache.set(key, embedding);
    }

    /** @internal */
    private _cacheAudioEmbedding(key: string, embedding: AudioEmbedding): void {
        if (this._audioCache.size >= this._maxAudioCache) {
            // Remove oldest entry (simple FIFO)
            const firstKey = this._audioCache.keys().next().value;
            this._audioCache.delete(firstKey);
        }
        this._audioCache.set(key, embedding);
    }

    /** @internal */
    private async _saveImageToTemp(input: ImageInput): Promise<string> {
        // TODO: Implement saving image data to temp file
        throw new Error("Image data input not yet implemented - please use file path");
    }

    /** @internal */
    private async _saveAudioToTemp(input: AudioInput): Promise<string> {
        // TODO: Implement saving audio data to temp file
        throw new Error("Audio data input not yet implemented - please use file path");
    }
}