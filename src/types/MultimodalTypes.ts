import {LlamaTextJSON} from "../utils/LlamaText.js";
import {GbnfJsonSchema, GbnfJsonSchemaToType} from "../utils/gbnfJson/types.js";

export type ImageInput = {
    /** Path to the image file on the filesystem */
    path: string;
    /** Optional custom identifier for the image */
    id?: string;
    /** Optional description or caption for the image */
    description?: string;
} | {
    /** Base64 encoded image data */
    data: string;
    /** MIME type of the image (e.g., 'image/jpeg', 'image/png') */
    mimeType: string;
    /** Optional custom identifier for the image */
    id?: string;
    /** Optional description or caption for the image */
    description?: string;
} | {
    /** Buffer containing image data */
    buffer: Uint8Array;
    /** MIME type of the image (e.g., 'image/jpeg', 'image/png') */
    mimeType: string;
    /** Optional custom identifier for the image */
    id?: string;
    /** Optional description or caption for the image */
    description?: string;
};

export type AudioInput = {
    /** Path to the audio file on the filesystem */
    path: string;
    /** Optional custom identifier for the audio */
    id?: string;
    /** Optional description or transcript for the audio */
    description?: string;
    /** Audio processing options */
    options?: AudioProcessingOptions;
} | {
    /** Base64 encoded audio data */
    data: string;
    /** MIME type of the audio (e.g., 'audio/wav', 'audio/mp3', 'audio/flac') */
    mimeType: string;
    /** Optional custom identifier for the audio */
    id?: string;
    /** Optional description or transcript for the audio */
    description?: string;
    /** Audio processing options */
    options?: AudioProcessingOptions;
} | {
    /** Buffer containing audio data */
    buffer: Uint8Array;
    /** MIME type of the audio (e.g., 'audio/wav', 'audio/mp3', 'audio/flac') */
    mimeType: string;
    /** Optional custom identifier for the audio */
    id?: string;
    /** Optional description or transcript for the audio */
    description?: string;
    /** Audio processing options */
    options?: AudioProcessingOptions;
};

export type AudioProcessingOptions = {
    /** Sample rate for audio processing (Hz) */
    sampleRate?: number;
    /** Number of channels (1 for mono, 2 for stereo) */
    channels?: number;
    /** Audio duration limit in seconds */
    maxDuration?: number;
    /** Whether to normalize audio levels */
    normalize?: boolean;
    /** Language hint for speech recognition */
    language?: string;
    /** Whether to generate transcript alongside embeddings */
    generateTranscript?: boolean;
};

export type ImageEmbedding = {
    /** The embedded vector representation of the image */
    embedding: Float32Array;
    /** Identifier for the image this embedding represents */
    imageId: string;
    /** Dimensions of the embedding vector */
    dimensions: number;
    /** Metadata about the image processing */
    metadata?: {
        /** Original image dimensions */
        originalWidth?: number;
        originalHeight?: number;
        /** Processing timestamp */
        processedAt?: Date;
        /** Model used for embedding */
        model?: string;
    };
};

export type AudioEmbedding = {
    /** The embedded vector representation of the audio */
    embedding: Float32Array;
    /** Identifier for the audio this embedding represents */
    audioId: string;
    /** Dimensions of the embedding vector */
    dimensions: number;
    /** Generated transcript if speech-to-text was enabled */
    transcript?: string;
    /** Metadata about the audio processing */
    metadata?: {
        /** Original audio duration in seconds */
        duration?: number;
        /** Sample rate used for processing */
        sampleRate?: number;
        /** Number of channels processed */
        channels?: number;
        /** Language detected/used */
        language?: string;
        /** Confidence score for transcript (0-1) */
        transcriptConfidence?: number;
        /** Processing timestamp */
        processedAt?: Date;
        /** Model used for embedding */
        model?: string;
    };
};

export type MultimodalInput = {
    /** Text input */
    text?: string;
    /** Image inputs */
    images?: ImageInput[];
    /** Audio inputs */
    audio?: AudioInput[];
    /** Pre-computed image embeddings (for performance optimization) */
    imageEmbeddings?: ImageEmbedding[];
    /** Pre-computed audio embeddings (for performance optimization) */
    audioEmbeddings?: AudioEmbedding[];
};

export type MultimodalChatMessage = {
    type: "user";
    content: MultimodalInput;
} | {
    type: "system";
    text: string | LlamaTextJSON;
} | {
    type: "model";
    response: Array<string | MultimodalChatModelFunctionCall | MultimodalChatModelSegment>;
};

export type MultimodalChatModelFunctionCall = {
    type: "functionCall";
    name: string;
    description?: string;
    params: any;
    result: any;
    rawCall?: LlamaTextJSON;
    startsNewChunk?: boolean;
};

export type MultimodalChatModelSegment = {
    type: "segment";
    segmentType: "thought" | "comment";
    text: string;
    ended: boolean;
    raw?: LlamaTextJSON;
    startTime?: string;
    endTime?: string;
};

export type MultimodalHistoryItem = MultimodalChatMessage;

export type MultimodalSessionModelFunctions = {
    readonly [name: string]: MultimodalSessionModelFunction<any>;
};

export type MultimodalSessionModelFunction<Params extends GbnfJsonSchema | undefined = GbnfJsonSchema | undefined> = {
    readonly description?: string;
    readonly params?: Params;
    readonly handler: (params: GbnfJsonSchemaToType<NoInfer<Params>>, context?: {
        images?: ImageEmbedding[];
        audio?: AudioEmbedding[];
    }) => any;
};

export type MultimodalCapabilities = {
    /** Vision capabilities */
    vision: {
        /** Whether vision processing is supported */
        supported: boolean;
        /** Maximum number of images that can be processed in a single inference */
        maxImages: number;
        /** Supported image formats */
        supportedFormats: string[];
        /** Maximum image resolution supported */
        maxResolution: {width: number; height: number};
        /** Whether the model supports image generation */
        supportsImageGeneration: boolean;
        /** Whether the model supports image understanding */
        supportsImageUnderstanding: boolean;
        /** Whether the model supports visual question answering */
        supportsVQA: boolean;
    };

    /** Audio capabilities */
    audio: {
        /** Whether audio processing is supported */
        supported: boolean;
        /** Maximum number of audio files that can be processed in a single inference */
        maxAudioFiles: number;
        /** Supported audio formats */
        supportedFormats: string[];
        /** Maximum audio duration in seconds */
        maxDuration: number;
        /** Supported sample rates */
        supportedSampleRates: number[];
        /** Whether the model supports speech-to-text */
        supportsSpeechToText: boolean;
        /** Whether the model supports audio understanding */
        supportsAudioUnderstanding: boolean;
        /** Whether the model supports audio generation */
        supportsAudioGeneration: boolean;
        /** Supported languages for speech recognition */
        supportedLanguages?: string[];
    };
};

// Keep the old type for backward compatibility
export type VisionModelCapabilities = MultimodalCapabilities["vision"];