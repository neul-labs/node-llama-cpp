import {DisposedError, EventRelay} from "lifecycle-utils";
import {LlamaMultimodalContext} from "../LlamaMultimodalContext/LlamaMultimodalContext.js";
import {LlamaMultimodalModel} from "../LlamaMultimodalModel/LlamaMultimodalModel.js";
import {LlamaText} from "../../utils/LlamaText.js";
import {
    ImageInput, AudioInput, MultimodalChatMessage, MultimodalHistoryItem,
    MultimodalSessionModelFunctions, MultimodalSessionModelFunction
} from "../../types/MultimodalTypes.js";

export type LlamaMultimodalChatSessionOptions = {
    /** The multimodal context to use for the chat session */
    context: LlamaMultimodalContext;

    /** System message to set the assistant's behavior */
    systemPrompt?: string;

    /** Maximum number of tokens to generate */
    maxTokens?: number;

    /** Temperature for randomness (0-1) */
    temperature?: number;

    /** Top-p sampling parameter */
    topP?: number;

    /** Top-k sampling parameter */
    topK?: number;

    /** Whether to automatically process images when added */
    autoProcessImages?: boolean;

    /** Whether to automatically process audio when added */
    autoProcessAudio?: boolean;

    /** Function calling capabilities */
    functions?: MultimodalSessionModelFunctions;

    /** Custom prompt template for multimodal interactions */
    promptTemplate?: {
        system?: string;
        user?: string;
        assistant?: string;
        imageMarker?: string;
        audioMarker?: string;
    };
};

export type LlamaMultimodalPromptOptions = {
    /** Maximum tokens to generate */
    maxTokens?: number;

    /** Temperature for randomness */
    temperature?: number;

    /** Top-p sampling */
    topP?: number;

    /** Top-k sampling */
    topK?: number;

    /** Stop sequences */
    stop?: string[];

    /** Whether to trim whitespace from response */
    trimWhitespaceSuffix?: boolean;

    /** Custom functions to call */
    functions?: MultimodalSessionModelFunctions;

    /** Signal to abort generation */
    signal?: AbortSignal;
};

/**
 * LlamaMultimodalChatSession provides a chat session that can handle text, images, and audio
 * with the same level of usability as regular chat sessions.
 */
export class LlamaMultimodalChatSession {
    /** @internal */ private readonly _context: LlamaMultimodalContext;
    /** @internal */ private readonly _model: LlamaMultimodalModel;
    /** @internal */ private readonly _history: MultimodalHistoryItem[] = [];
    /** @internal */ private readonly _systemPrompt?: string;
    /** @internal */ private readonly _autoProcessImages: boolean;
    /** @internal */ private readonly _autoProcessAudio: boolean;
    /** @internal */ private readonly _functions?: MultimodalSessionModelFunctions;
    /** @internal */ private readonly _promptTemplate: Required<NonNullable<LlamaMultimodalChatSessionOptions['promptTemplate']>>;
    /** @internal */ private _disposed = false;

    public readonly onDispose = new EventRelay<void>();

    private constructor(options: LlamaMultimodalChatSessionOptions) {
        this._context = options.context;
        this._model = options.context.model as LlamaMultimodalModel;
        this._systemPrompt = options.systemPrompt;
        this._autoProcessImages = options.autoProcessImages ?? true;
        this._autoProcessAudio = options.autoProcessAudio ?? true;
        this._functions = options.functions;

        this._promptTemplate = {
            system: options.promptTemplate?.system ?? "System: {content}",
            user: options.promptTemplate?.user ?? "User: {content}",
            assistant: options.promptTemplate?.assistant ?? "Assistant: {content}",
            imageMarker: options.promptTemplate?.imageMarker ?? "<image>",
            audioMarker: options.promptTemplate?.audioMarker ?? "<audio>"
        };

        // Listen for context disposal
        this._context.onDispose.createListener(() => {
            void this.dispose();
        });

        // Add system prompt to history if provided
        if (this._systemPrompt) {
            this._history.push({
                type: "system",
                text: this._systemPrompt,
                timestamp: Date.now()
            });
        }
    }

    /**
     * Create a new LlamaMultimodalChatSession instance
     */
    public static async create(options: LlamaMultimodalChatSessionOptions): Promise<LlamaMultimodalChatSession> {
        return new LlamaMultimodalChatSession(options);
    }

    /**
     * Send a multimodal prompt and get a response
     */
    public async prompt(
        message: MultimodalChatMessage,
        options?: LlamaMultimodalPromptOptions
    ): Promise<string> {
        this._ensureNotDisposed();

        // Add user message to history
        const historyItem: MultimodalHistoryItem = {
            type: "user",
            text: typeof message === "string" ? message : message.text || "",
            timestamp: Date.now()
        };

        // Process images if present
        if (typeof message !== "string" && message.images) {
            historyItem.images = [];
            for (const image of message.images) {
                if (this._autoProcessImages) {
                    const embedding = await this._context.addImage(image);
                    historyItem.images.push({ input: image, embedding });
                } else {
                    historyItem.images.push({ input: image });
                }
            }
        }

        // Process audio if present
        if (typeof message !== "string" && message.audio) {
            historyItem.audio = [];
            for (const audio of message.audio) {
                if (this._autoProcessAudio) {
                    const embedding = await this._context.addAudio(audio, {
                        generateTranscript: true
                    });
                    historyItem.audio.push({ input: audio, embedding });
                } else {
                    historyItem.audio.push({ input: audio });
                }
            }
        }

        this._history.push(historyItem);

        // Build the prompt
        const prompt = this._buildPrompt();

        // Generate response using base context
        const sequence = this._context.getSequence();

        try {
            const response = await sequence.generate(new LlamaText(prompt), {
                maxTokens: options?.maxTokens ?? 1000,
                temperature: options?.temperature ?? 0.7,
                topP: options?.topP ?? 0.9,
                topK: options?.topK ?? 40,
                stopOnAbortSignal: true,
                signal: options?.signal,
                onToken: (tokens) => {
                    // Could emit token events here if needed
                }
            });

            const responseText = this._model.tokenizer.detokenize(response);

            // Add assistant response to history
            this._history.push({
                type: "assistant",
                text: responseText,
                timestamp: Date.now()
            });

            return responseText;

        } finally {
            // Cleanup sequence resources if needed
        }
    }

    /**
     * Add an image to the current conversation context
     */
    public async addImage(image: ImageInput): Promise<void> {
        this._ensureNotDisposed();
        await this._context.addImage(image);
    }

    /**
     * Add audio to the current conversation context
     */
    public async addAudio(audio: AudioInput, options?: {
        generateTranscript?: boolean;
        language?: string;
    }): Promise<void> {
        this._ensureNotDisposed();
        await this._context.addAudio(audio, options);
    }

    /**
     * Get the conversation history
     */
    public getHistory(): readonly MultimodalHistoryItem[] {
        return this._history;
    }

    /**
     * Clear the conversation history (but keep system prompt)
     */
    public clearHistory(): void {
        this._ensureNotDisposed();

        // Keep only system prompt
        const systemPrompts = this._history.filter(item => item.type === "system");
        this._history.length = 0;
        this._history.push(...systemPrompts);

        // Clear multimodal context
        this._context.clearMultimodalContent();
    }

    /**
     * Get currently active images in the conversation
     */
    public get activeImages() {
        return this._context.activeImages;
    }

    /**
     * Get currently active audio in the conversation
     */
    public get activeAudio() {
        return this._context.activeAudio;
    }

    /**
     * Get the underlying context
     */
    public get context(): LlamaMultimodalContext {
        return this._context;
    }

    /**
     * Get the underlying model
     */
    public get model(): LlamaMultimodalModel {
        return this._model;
    }

    /**
     * Check if the session is disposed
     */
    public get isDisposed(): boolean {
        return this._disposed;
    }

    /**
     * Dispose of the chat session and free resources
     */
    public async dispose(): Promise<void> {
        if (this._disposed) {
            return;
        }

        this._disposed = true;
        this._history.length = 0;

        // Note: We don't dispose the context here as it might be shared
        this.onDispose.dispatchEvent();
    }

    /** @internal */
    private _ensureNotDisposed(): void {
        if (this._disposed) {
            throw new DisposedError();
        }
    }

    /** @internal */
    private _buildPrompt(): string {
        const parts: string[] = [];

        for (const item of this._history) {
            let template: string;
            switch (item.type) {
                case "system":
                    template = this._promptTemplate.system;
                    break;
                case "user":
                    template = this._promptTemplate.user;
                    break;
                case "assistant":
                    template = this._promptTemplate.assistant;
                    break;
                default:
                    continue;
            }

            let content = item.text;

            // Add image markers
            if (item.images && item.images.length > 0) {
                const imageMarkers = item.images.map(() => this._promptTemplate.imageMarker).join(" ");
                content = `${imageMarkers} ${content}`;
            }

            // Add audio markers and transcripts
            if (item.audio && item.audio.length > 0) {
                const audioContent = item.audio.map(a => {
                    let audioText = this._promptTemplate.audioMarker;
                    if (a.embedding?.transcript) {
                        audioText += ` [Transcript: ${a.embedding.transcript}]`;
                    }
                    return audioText;
                }).join(" ");
                content = `${audioContent} ${content}`;
            }

            parts.push(template.replace("{content}", content));
        }

        // Add assistant prompt starter
        parts.push(this._promptTemplate.assistant.replace("{content}", ""));

        return parts.join("\\n\\n");
    }
}