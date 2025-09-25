# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Building and Development
```bash
# Install dependencies
npm install

# Build TypeScript
npm run build

# Build and pack templates
npm run build:packTemplates

# Watch mode for development
npm run watch

# Full development setup (includes downloading test models)
npm run dev:setup

# Build llama.cpp from source
npm run cmake-js-llama
```

### Testing
```bash
# Run all tests
npm run test

# TypeScript compilation check
npm run test:typescript

# Individual test suites
npm run test:vitest         # Core tests
npm run test:standalone     # Standalone tests
npm run test:modelDependent # Tests requiring models

# Interactive test modes
npm run test:standalone:interactive
npm run test:modelDependent:interactive
```

### Linting and Formatting
```bash
# Lint code
npm run lint

# Fix linting issues
npm run format
```

### Documentation
```bash
# Generate API docs and start dev server
npm run docs:dev

# Build documentation
npm run docs:build

# Preview built docs
npm run docs:preview
```

### Building Native Components
The project includes native C++ bindings for llama.cpp integration:

```bash
# Build the native addon (includes multimodal support if available)
cd llama && cmake-js

# Clean rebuild
npm run clean && npm install
```

## Architecture Overview

### Core Components

**Bindings Layer** (`src/bindings/`)
- `Llama.ts` - Main llama.cpp binding interface
- `getLlama.ts` - Factory function for Llama instances
- `AddonTypes.ts` - TypeScript definitions for C++ addon functions
- Handles GPU detection, binary management, and native library loading

**Evaluator Layer** (`src/evaluator/`)
- `LlamaModel/` - Model loading and management
- `LlamaContext/` - Context and sequence management for inference
- `LlamaChatSession/` - High-level chat interface
- `LlamaCompletion.ts` - Text completion functionality
- `LlamaEmbeddingContext.ts` - Embedding generation
- `LlamaMultimodal*/` - Multimodal model support (images, audio)

**Chat Wrappers** (`src/chatWrappers/`)
- Model-specific prompt formatting (LLaMA, Qwen, Mistral, etc.)
- Template-based and Jinja template support
- Automatic wrapper detection based on model metadata

**Native Integration** (`llama/`)
- `addon/` - C++ Node.js addon code
- `CMakeLists.txt` - Build configuration with multimodal support
- `llama.cpp/` - Submodule of the llama.cpp project

### Key Architectural Patterns

**Resource Management**
- All major classes implement disposal pattern with `onDispose` events
- Resources must be explicitly disposed to free memory
- Lifecycle management through `lifecycle-utils` library

**Type Safety**
- Comprehensive TypeScript definitions throughout
- Runtime type validation for critical paths
- Strict null checks and type guards

**Multimodal Support**
The project includes comprehensive multimodal capabilities:
- `LlamaMultimodalModel` - Extends base models with vision/audio
- `LlamaMultimodalContext` - Context with image/audio processing
- `LlamaMultimodalChatSession` - Chat sessions with media support
- Native C++ bindings with mtmd library integration
- Intelligent caching for embeddings

### File Structure Patterns

**TypeScript Sources** (`src/`)
- Entry point: `index.ts` - Re-exports all public APIs
- Each major component in its own directory
- Types defined alongside implementation or in dedicated `types/` directories

**Native Code** (`llama/addon/`)
- `addon.cpp` - Main N-API bindings
- Conditional compilation for multimodal features
- Integration with llama.cpp's C API

**Documentation** (`docs/`)
- VitePress-based documentation system
- Guide pages in `docs/guide/`
- API documentation auto-generated from TypeScript

## Testing Architecture

**Test Categories**
- **Standalone tests** - No model dependencies, fast execution
- **Model-dependent tests** - Require downloaded models, slower
- **TypeScript tests** - Compilation and type checking

**Model Management**
- Test models stored in `test/.models/`
- Models downloaded via `npm run dev:setup`
- Different model sizes for different test scenarios

## Build System

**TypeScript Compilation**
- ESM-only output with strict type checking
- Composite project structure
- Source maps and declarations generated

**Native Compilation**
- CMake-based build for llama.cpp integration
- Conditional features based on available libraries
- Cross-platform support (macOS, Linux, Windows)
- CUDA, Metal, and Vulkan backend support

**Multimodal Build**
The build system automatically detects and enables multimodal support:
- Checks for mtmd library in llama.cpp
- Enables CLIP vision and Whisper audio processing
- Conditional compilation via `LLAMA_MTMD_AVAILABLE`

## Key Development Considerations

**Memory Management**
- Explicit disposal required for all Llama-related objects
- GPU memory allocation handled automatically based on available VRAM
- Context size and batch size impact memory usage significantly

**Model Loading**
- Models loaded via file paths or URLs
- Automatic model format detection (GGUF)
- Model metadata extraction and validation
- Chat wrapper auto-detection from model metadata

**Performance**
- Batching support for multiple parallel sequences
- GPU acceleration with automatic backend selection
- Token prediction and speculative decoding
- Caching mechanisms for embeddings and processed media

**Multimodal Integration**
- Images and audio processed through native bindings
- Embedding caching with LRU eviction
- Support for multiple model architectures (LLaVA, Qwen2-VL, LFM2-VL)
- Same API patterns as text-only functionality

## Error Handling Patterns

**Disposal Errors**
- `DisposedError` thrown when using disposed objects
- Check `isDisposed` property before operations

**Model Loading Errors**
- File not found errors with helpful messages
- GPU memory insufficient errors with suggestions
- Model format validation errors

**Build Errors**
- Native compilation failures with troubleshooting guidance
- Missing dependency detection and installation suggestions