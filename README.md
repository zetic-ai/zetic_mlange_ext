<!---
Copyright 2025 ZETIC.ai Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# ZETIC MLange Extension (zetic_mlange_ext)

<p align="center">
    <a href="https://docs.zetic.ai"><img alt="MLange Documentation" src="https://img.shields.io/endpoint?url=https://docs.zetic.ai&color=brightgreen"></a>
    <a href="https://mlange.zetic.ai"><img alt="LICENSE" src="https://img.shields.io/circleci/build/github/huggingface/transformers/main"></a>
    <a href="https://github.com/huggingface/transformers/blob/main/LICENSE"><img alt="MLange Dashboard" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue"></a>
    <a href="https://huggingface.co/docs/transformers/index"><img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online"></a>
    <a href="https://github.com/huggingface/transformers/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/transformers.svg"></a>
    <a href="https://github.com/huggingface/transformers/blob/main/CODE_OF_CONDUCT.md"><img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg"></a>
</p>


<h3 align="center">
    <p>A comprehensive AI extension library for Edge AI runtime environments</p>
</h3>

 ZETIC MLange Extension is a cross-platform extension library for AI runtime environments, providing ready-to-use mobile deployment solutions with seamless integration of input/output processing capabilities and neural network inference pipelines built on the ZETIC.MLange framework.

 ZETIC MLange Extension bridges the gap between AI research and mobile production deployment by offering a unified, cross-platform interface for common preprocessing, postprocessing, and inference operations. Our core C/C++ implementation ensures optimal performance while providing native bindings for Edge AI environments, making it easy to deploy AI pipelines directly on edge devices.


## Key Features

### 📱 Mobile-First Architecture

- Native Android Support: Ready-to-use Kotlin/Java packages for Android development
- Native iOS Support: Swift packages optimized for iOS deployment
- Cross-Platform Support: Flutter, React-Native packages
- Cross-Platform Core: High-performance C/C++ implementation with platform-specific bindings
- Device Optimization: Memory and compute optimizations for mobile hardware constraints

### 🚀 Neural Network Inference Pipeline

- ZETIC.MLange Integration: Native support for ZETIC.MLange framework inference
- Pipeline Orchestration: Configurable multi-stage processing pipelines

### ⚡ Performance & Scalability

- Memory Optimization: Efficient memory management for resource-constrained environments
- Caching: Intelligent caching for repeated operations

      
## Installation

### 1. Import to your mobile project
#### Gradle
```kotlin
implementation("com.zeticai:mlange:ext:0.0.1")
```
#### SPM
```
github.com/zetic-ai/zetic_mlange_ext.git
```

### 2. Build From source code

### Prerequisite
- java19
- NDK ($ANDROID_NDK)

### Build
1. Build `third-party` libraries by run `build_third-party.sh`.
2. Build android or ios project using Android Studio or Xcode.


## Quick Start

- Android(Kotlin)

``` kotlin
val objectDetectionPipeline = ZeticMLangePipeline(
    feature = ObjectDetection(this),
    inputSource = CameraSource(this, surface.holder, preferredSize),
)

objectDetectionPipeline.loop { runOnUiThread {
    yolo.visualize(YOLOResult(it.value), preferredSize, true)
}}
```

- iOS(Swift)

``` swift
GeometryReader { geometry in
    ZStack {
        if let previewLayer = cameraSource.previewLayer {
            CameraPreviewView(previewLayer: previewLayer)
        }
        
        if let detections = pipeline.latestResult {
            DetectionsView(detectionResult: detections, cameraResolution: cameraSource.resolution)
        }
    }
}
.onAppear {
    pipeline.startLoop()
}
.onDisappear {
    pipeline.stopLoop()
}
```


## Why should I use ZETIC MLange Extension?

### 🚀 Production-Ready Mobile Deployment
  - ZETIC.MLange is specifically designed for real-world mobile applications, not just research prototypes. While other frameworks focus on training or server deployment, ZETIC.MLange excels at bringing AI models to resource-constrained mobile devices with consistent performance and reliability.

### ⚡ Unmatched Mobile Performance
  - Our C/C++ core delivers superior performance compared to Python-based alternatives or framework-specific mobile solutions. ZETIC.MLange leverages platform-specific optimizations (ARM NEON, GPU acceleration, NPU utilization) to maximize inference speed while minimizing battery consumption and memory usage.

### 🔧 Complete Pipeline Solution
  - Unlike other frameworks that only handle inference, ZETIC.MLange provides end-to-end pipeline management including preprocessing (OpenCV, tokenizers), inference, and postprocessing in a single, cohesive package. This eliminates the complexity of integrating multiple libraries and ensures optimal data flow between pipeline stages.

### 📱 True Cross-Platform Development
  - Write your AI pipeline once in C/C++ and deploy everywhere. ZETIC.MLange automatically generates native bindings for Android (Kotlin/Java), iOS (Swift) and Cross Platform(Flutter, React Native), ensuring your code feels natural on each platform while maintaining identical inference results and performance characteristics.

### 🛡️ Enterprise-Grade Reliability
  - ZETIC.MLange is built for production environments with robust error handling, memory management, and graceful degradation. Our architecture prevents common mobile AI pitfalls like memory leaks, crashes from large models, and inconsistent inference results across devices.

### 🎯 Developer-Friendly Design
  - Simple, intuitive APIs that hide complexity without sacrificing control. Whether you're a mobile developer new to AI or an ML engineer moving to mobile, ZETIC.MLange's clean interfaces and comprehensive documentation get you productive quickly.

### 💡 Extensible Architecture
  - Built with modularity in mind, ZETIC.MLange makes it easy to add custom processors, integrate new model formats, or extend functionality without modifying core components. The plugin architecture ensures your extensions work seamlessly across all supported platforms.


## Example Models

<details>
<summary>Object Detection</summary>

- [YOLOv8](https://huggingface.co/Ultralytics/YOLOv8)
- [YOLOv11](https://huggingface.co/Ultralytics/YOLO11)

</details>

<details>
<summary>Automatic Speech Recognition></summary>

- [Whisper](https://huggingface.co/openai/whisper-small)

</details>

<details>
<summary>Sound Classification</summary>

- [YAMNet](https://github.com/w-hc/torch_audioset)

</details>

<details>
<summary>Face Emotion Recognition</summary>

- [FER](https://huggingface.co/ElenaRyumina/face_emotion_recognition)

</details>



-------------------------------------------------------

## Built with ❤️ for mobile AI deployment by the community
