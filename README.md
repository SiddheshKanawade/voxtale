# VoxTale

**Automated faceless documentary video generator**

VoxTale transforms PDF documents into engaging documentary-style videos with AI-generated narration, visuals, and subtitles. Perfect for creating educational content, biographical videos, or documentary-style presentations without manual video editing.

🎥 **See VoxTale in action**: [**@authenticvoxtale**](https://www.youtube.com/@authenticvoxtale)

## What it does

- **PDF to Script**: Extracts content from PDF files and generates documentary-style scripts using GPT
- **AI Narration**: Converts scripts to natural-sounding speech using ElevenLabs TTS
- **Visual Generation**: Creates contextual images using FLUX AI image generation
- **Video Assembly**: Combines narration, visuals, and subtitles into polished videos with Ken Burns effects
- **Background Audio**: Adds ambient background music with automatic volume balancing
- **Subtitle Generation**: Creates accurate word-level subtitles using Whisper transcription

## Key Features

- Fully automated video production pipeline
- Support for long-form content (6000+ character scripts)
- Dynamic image transitions and zoom effects
- Professional subtitle styling with semi-transparent backgrounds
- Background music integration with looping and volume control
- FastAPI service for programmatic video generation
- Optional cloud storage integration (R2/S3 compatible)

## Technology Stack

- **AI Services**: OpenAI GPT, ElevenLabs TTS, Together AI (FLUX), OpenAI Whisper
- **Video Processing**: MoviePy for video assembly and effects
- **Backend**: FastAPI for API service
- **Storage**: Local filesystem with optional cloud upload 