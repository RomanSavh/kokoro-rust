# kokoro-rust

A Rust library for high-quality, real-time Text-to-Speech (TTS) using the Kokoro model. This library is based on [Kokoros](https://github.com/lucasjinreal/Kokoros), but provides a library interface for easy integration into Rust applications.

## Overview

`kokoro-rust` is a library wrapper around the Kokoro TTS model, providing a clean and efficient API for generating speech from text. It supports:

- **High-quality TTS**: Based on the [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) model
- **Word-level timestamps**: Get precise timing information for each word in the generated audio
- **Streaming support**: Generate audio in real-time chunks for low-latency applications
- **Multiple voices**: Support for various voice styles
- **ONNX Runtime**: Efficient inference using ONNX Runtime with optional CUDA support
- **Language support**: Multiple languages including English, Mandarin, Japanese, and more

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
kokoro-rust = "0.1.0"
```

For CUDA support (if you have a compatible GPU):

```toml
[dependencies]
kokoro-rust = { version = "0.1.0", features = ["cuda"] }
```

## Quick Start

### Basic Usage

```rust
use kokoro_rust::tts::koko::{TTSKoko, InitConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the TTS engine
    let config = InitConfig {
        model_path: "checkpoints/kokoro-v1.0.onnx".to_string(),
        voices_path: "data/voices-v1.0.bin".to_string(),
        sample_rate: 24000,
    };
    
    let tts = TTSKoko::from_config(config).await;
    
    // Generate speech
    let opts = kokoro_rust::tts::koko::TTSOpts {
        txt: "Hello, this is a test of the Kokoro TTS system!",
        lan: "en",
        style_name: "af_sky",
        save_path: "output.wav",
        mono: true,
        speed: 1.0,
        initial_silence: None,
    };
    
    tts.tts(&opts).await?;
    
    Ok(())
}
```

### Generate Audio with Word Timestamps

```rust
use kokoro_rust::tts::koko::{TTSKoko, InitConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = InitConfig {
        model_path: "checkpoints/kokoro-v1.0.onnx".to_string(),
        voices_path: "data/voices-v1.0.bin".to_string(),
        sample_rate: 24000,
    };
    
    let tts = TTSKoko::from_config(config).await;
    
    // Generate audio with word-level timestamps
    let (audio, alignments) = tts.tts_timestamped_raw_audio(
        "Hello from the timestamped model",
        "en",
        "af_sky",
        true,
        1.0,
        None,
    ).await?;
    
    // audio: Vec<f32> - Raw audio samples
    // alignments: Vec<WordAlignment> - Word timing information
    //   Each WordAlignment contains: word, start_sec, end_sec
    
    Ok(())
}
```

### Streaming Audio Generation

```rust
use kokoro_rust::tts::koko::{TTSKoko, InitConfig, TtsOutput};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = InitConfig {
        model_path: "checkpoints/kokoro-v1.0.onnx".to_string(),
        voices_path: "data/voices-v1.0.bin".to_string(),
        sample_rate: 24000,
    };
    
    let tts = TTSKoko::from_config(config).await;
    
    // Stream audio chunks as they're generated
    tts.tts_raw_audio_streaming(
        "This is a streaming test with real-time audio generation.",
        "en",
        "af_sky",
        true,
        1.0,
        None,
        |output: TtsOutput| -> Result<(), Box<dyn std::error::Error>> {
            match output {
                TtsOutput::Audio(audio) => {
                    // Process audio chunk immediately
                    println!("Received {} samples", audio.len());
                }
                TtsOutput::Aligned(audio, alignments) => {
                    // Process audio with timestamps
                    println!("Received {} samples with {} word alignments", 
                             audio.len(), alignments.len());
                }
            }
            Ok(())
        }
    ).await?;
    
    Ok(())
}
```

## API Reference

### `TTSKoko`

The main TTS engine struct.

#### Methods

- **`from_config(config: InitConfig) -> Self`**: Initialize the TTS engine with configuration
- **`tts(opts: &TTSOpts) -> Result<(), Box<dyn Error>>`**: Generate speech and save to file
- **`tts_raw_audio(...) -> Result<Vec<f32>, Box<dyn Error>>`**: Generate raw audio samples
- **`tts_timestamped_raw_audio(...) -> Result<(Vec<f32>, Vec<WordAlignment>), Box<dyn Error>>`**: Generate audio with word timestamps
- **`tts_raw_audio_streaming<F>(..., callback: F) -> Result<(), Box<dyn Error>>`**: Stream audio chunks
- **`tts_timestamped_raw_audio_streaming<F>(..., callback: F) -> Result<(), Box<dyn Error>>`**: Stream audio with timestamps
- **`get_available_voices(&self) -> Vec<String>`**: Get list of available voice styles
- **`mix_styles(&self, style_names: &[&str], weights: &[f32]) -> Result<Vec<[[f32; 256]; 1]>, Box<dyn Error>>`**: Mix multiple voice styles

### `InitConfig`

Configuration for initializing the TTS engine.

```rust
pub struct InitConfig {
    pub model_path: String,      // Path to ONNX model file
    pub voices_path: String,      // Path to voices binary file
    pub sample_rate: u32,         // Audio sample rate (typically 24000)
}
```

### `TTSOpts`

Options for text-to-speech generation.

```rust
pub struct TTSOpts<'a> {
    pub txt: &'a str,             // Text to synthesize
    pub lan: &'a str,             // Language code (e.g., "en", "zh", "ja")
    pub style_name: &'a str,      // Voice style name (e.g., "af_sky")
    pub save_path: &'a str,        // Output file path
    pub mono: bool,               // Mono audio output
    pub speed: f32,               // Speech speed multiplier
    pub initial_silence: Option<usize>, // Initial silence in samples
}
```

### `WordAlignment`

Word-level timing information.

```rust
pub struct WordAlignment {
    pub word: String,      // The word text
    pub start_sec: f32,    // Start time in seconds
    pub end_sec: f32,      // End time in seconds
}
```

### `TtsOutput`

Output from streaming TTS operations.

```rust
pub enum TtsOutput {
    Audio(Vec<f32>),                                    // Audio samples only
    Aligned(Vec<f32>, Vec<WordAlignment>),             // Audio with timestamps
}
```

## Model Files

You'll need to download the model files before using the library:

1. **ONNX Model**: Download from [Hugging Face](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX-timestamped) or the original [Kokoros repository](https://github.com/lucasjinreal/Kokoros)
2. **Voices Data**: Download `voices-v1.0.bin` from the [kokoro-onnx releases](https://github.com/thewh1teagle/kokoro-onnx/releases)

Example download commands:

```bash
# Create directories
mkdir -p checkpoints data

# Download ONNX model
curl -L \
  "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX-timestamped/resolve/main/onnx/model.onnx" \
  -o checkpoints/kokoro-v1.0.onnx

# Download voices data
curl -L \
  "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin" \
  -o data/voices-v1.0.bin
```

## Features

- **`default`**: CPU-based inference (enabled by default)
- **`cuda`**: CUDA/GPU acceleration support (requires CUDA-compatible GPU)

## Performance

The library is designed for high-performance TTS generation:

- **Low latency**: Streaming support for real-time applications
- **Efficient**: Uses ONNX Runtime for optimized inference
- **Thread-safe**: Safe for use in concurrent applications
- **Memory efficient**: Streaming mode reduces memory footprint

## Differences from Kokoros

This library (`kokoro-rust`) is based on [Kokoros](https://github.com/lucasjinreal/Kokoros) but provides:

- **Library interface**: Designed to be used as a dependency in other Rust projects
- **API-focused**: Clean, documented API for programmatic use
- **No CLI**: Focuses on library functionality rather than command-line tools
- **Modular**: Well-structured modules for easy integration

## License

This project is licensed under the Apache License 2.0.

## Acknowledgments

- Based on [Kokoros](https://github.com/lucasjinreal/Kokoros) by Lucas Jin
- Uses the [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) model
- ONNX model conversion by [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please open an issue on the GitHub repository.

