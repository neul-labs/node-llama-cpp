import {DisposedError, EventRelay} from "lifecycle-utils";
import type {LlamaContext, LlamaContextOptions} from "../LlamaContext/LlamaContext.js";
import {LlamaMultimodalModel} from "../LlamaMultimodalModel/LlamaMultimodalModel.js";
import {ImageInput, AudioInput, ImageEmbedding, AudioEmbedding, MultimodalInput} from "../../types/MultimodalTypes.js";
import {LlamaText} from "../../utils/LlamaText.js";
import {Token} from "../../types.js";

export type LlamaMultimodalContextOptions = LlamaContextOptions & {
    /** Maximum number of images to keep in context */
    maxImagesInContext?: number;

    /** Maximum number of audio files to keep in context */
    maxAudioInContext?: number;

    /** Whether to automatically cache image embeddings */
    autoImageCache?: boolean;

    /** Whether to automatically cache audio embeddings */
    autoAudioCache?: boolean;

    /** Image preprocessing options */
    imagePreprocessing?: {
        /** Resize images to this resolution before processing */
        targetResolution?: {width: number; height: number};
        /** Image quality (0-1) for JPEG compression */
        quality?: number;
        /** Whether to maintain aspect ratio when resizing */
        maintainAspectRatio?: boolean;
    };

    /** Audio preprocessing options */
    audioPreprocessing?: {
        /** Target sample rate for audio processing */
        targetSampleRate?: number;
        /** Whether to normalize audio volume */
        normalizeVolume?: boolean;
        /** Maximum audio duration in seconds */
        maxDuration?: number;
    };
};

/**
 * LlamaMultimodalContext provides a context for multimodal text generation that can process
 * images and audio alongside text with the same level of usability as other API calls.
 */
export class LlamaMultimodalContext {
    /** @internal */ private readonly _baseContext: LlamaContext;
    /** @internal */ private readonly _model: LlamaMultimodalModel;
    /** @internal */ private readonly _activeImageEmbeddings: ImageEmbedding[] = [];
    /** @internal */ private readonly _activeAudioEmbeddings: AudioEmbedding[] = [];
    /** @internal */ private readonly _maxImagesInContext: number;
    /** @internal */ private readonly _maxAudioInContext: number;
    /** @internal */ private readonly _autoImageCache: boolean;
    /** @internal */ private readonly _autoAudioCache: boolean;
    /** @internal */ private readonly _imagePreprocessing?: LlamaMultimodalContextOptions['imagePreprocessing'];
    /** @internal */ private readonly _audioPreprocessing?: LlamaMultimodalContextOptions['audioPreprocessing'];
    /** @internal */ private _disposed = false;

    public readonly onDispose = new EventRelay<void>();

    private constructor(baseContext: LlamaContext, model: LlamaMultimodalModel, options: LlamaMultimodalContextOptions) {
        this._baseContext = baseContext;
        this._model = model;
        this._maxImagesInContext = options.maxImagesInContext ?? 4;
        this._maxAudioInContext = options.maxAudioInContext ?? 2;
        this._autoImageCache = options.autoImageCache ?? true;
        this._autoAudioCache = options.autoAudioCache ?? true;
        this._imagePreprocessing = options.imagePreprocessing;
        this._audioPreprocessing = options.audioPreprocessing;

        // Listen for base context disposal
        this._baseContext.onDispose.createListener(() => {
            void this.dispose();
        });
    }

    /**
     * Create a new LlamaMultimodalContext instance
     */
    public static async create(model: LlamaMultimodalModel, options: LlamaMultimodalContextOptions): Promise<LlamaMultimodalContext> {
        // Create the base text context first
        const baseContext = await model.createContext(options) as LlamaContext;

        // Create the multimodal wrapper
        return new LlamaMultimodalContext(baseContext, model, options);
    }

    /**
     * Add an image to the current context
     */
    public async addImage(input: ImageInput): Promise<ImageEmbedding> {
        this._ensureNotDisposed();

        const embedding = await this._model.processImage(input);

        // Manage context memory
        if (this._activeImageEmbeddings.length >= this._maxImagesInContext) {
            this._activeImageEmbeddings.shift(); // Remove oldest
        }

        this._activeImageEmbeddings.push(embedding);
        return embedding;
    }

    /**
     * Add audio to the current context
     */
    public async addAudio(input: AudioInput, options?: {
        generateTranscript?: boolean;
        language?: string;
    }): Promise<AudioEmbedding> {
        this._ensureNotDisposed();

        const embedding = await this._model.processAudio(input, options);

        // Manage context memory
        if (this._activeAudioEmbeddings.length >= this._maxAudioInContext) {
            this._activeAudioEmbeddings.shift(); // Remove oldest
        }

        this._activeAudioEmbeddings.push(embedding);
        return embedding;
    }

    /**
     * Evaluate a sequence of multimodal inputs (text, images, audio)
     */
    public async evaluate(inputs: MultimodalInput[], options?: {
        temperature?: number;
        topP?: number;
        topK?: number;
    }): Promise<{
        tokens: Token[];
        contextWindowPointer: number;
    }> {
        this._ensureNotDisposed();

        // Process multimodal inputs first
        const processedInputs: Token[] = [];

        for (const input of inputs) {
            if (typeof input === 'string' || input instanceof LlamaText || Array.isArray(input)) {
                // Text input - tokenize normally
                const tokens = this._tokenizeInput(input);
                processedInputs.push(...tokens);
            } else if ('path' in input || 'data' in input) {
                // Check if it's an image or audio input
                if (this._isImageInput(input)) {
                    const embedding = await this.addImage(input as ImageInput);
                    // Convert embedding to tokens (this would need proper implementation)
                    const tokens = this._embeddingToTokens(embedding.data);
                    processedInputs.push(...tokens);
                } else if (this._isAudioInput(input)) {
                    const embedding = await this.addAudio(input as AudioInput);
                    // Convert embedding to tokens (this would need proper implementation)
                    const tokens = this._embeddingToTokens(embedding.data);
                    processedInputs.push(...tokens);
                }
            }
        }

        // Delegate to base context for actual evaluation
        return await this._baseContext.evaluate(processedInputs, options as any);
    }

    /**
     * Clear all active multimodal content from context
     */
    public clearMultimodalContent(): void {
        this._ensureNotDisposed();
        this._activeImageEmbeddings.length = 0;
        this._activeAudioEmbeddings.length = 0;
    }

    /**
     * Get currently active images in context
     */
    public get activeImages(): readonly ImageEmbedding[] {
        return this._activeImageEmbeddings;
    }

    /**
     * Get currently active audio in context
     */
    public get activeAudio(): readonly AudioEmbedding[] {
        return this._activeAudioEmbeddings;
    }

    // Delegate base context properties and methods
    public get contextSize(): number { return this._baseContext.contextSize; }
    public get batchSize(): number { return this._baseContext.batchSize; }
    public get model() { return this._model; }
    public get isDisposed(): boolean { return this._disposed || this._baseContext.isDisposed; }

    /**
     * Get a sequence for text generation
     */
    public getSequence() {
        this._ensureNotDisposed();
        return this._baseContext.getSequence();
    }

    /**
     * Dispose of the multimodal context and free resources
     */
    public async dispose(): Promise<void> {
        if (this._disposed) {
            return;
        }

        this._disposed = true;
        this.clearMultimodalContent();

        await this._baseContext.dispose();
        this.onDispose.dispatchEvent();
    }

    /** @internal */
    private _ensureNotDisposed(): void {
        if (this._disposed) {
            throw new DisposedError();
        }
    }

    /** @internal */
    private _tokenizeInput(input: string | LlamaText | Token[]): Token[] {
        if (Array.isArray(input)) {
            return input;
        }

        if (typeof input === 'string') {
            return this._model.tokenizer.encode(input);
        }

        // Handle LlamaText
        return this._model.tokenizer.encode(input.toString());
    }

    /** @internal */
    private _isImageInput(input: any): boolean {
        return input && (
            input.format?.startsWith('image/') ||
            (typeof input === 'string' && /\.(jpg|jpeg|png|webp|bmp|gif)$/i.test(input)) ||
            (input.path && /\.(jpg|jpeg|png|webp|bmp|gif)$/i.test(input.path))
        );
    }

    /** @internal */
    private _isAudioInput(input: any): boolean {
        return input && (
            input.format?.startsWith('audio/') ||
            (typeof input === 'string' && /\.(wav|mp3|flac|ogg|m4a)$/i.test(input)) ||
            (input.path && /\.(wav|mp3|flac|ogg|m4a)$/i.test(input.path))
        );
    }

    /** @internal */
    private _embeddingToTokens(embedding: number[]): Token[] {
        // This would need proper implementation to convert embeddings to tokens
        // For now, return a placeholder token
        return [this._model.tokenizer.encode('<|vision|>')[0] || 0];
    }
}