# DeepFace-Recognition-Clustering

## 🔍 Advanced Face Recognition and Clustering System

A GPU-accelerated face recognition and clustering system that combines YOLO-based detection, VGG-Face embeddings, and advanced clustering techniques. Built to handle large-scale image datasets with comprehensive attribute analysis.

[![Python Version](https://img.shields.io/badge/python-3.6%2B-blue.svg)]()
[![CUDA Support](https://img.shields.io/badge/CUDA-enabled-green.svg)]()
[![DeepFace](https://img.shields.io/badge/DeepFace-integrated-orange.svg)]()
[![YOLOv8](https://img.shields.io/badge/YOLOv8-detection-yellow.svg)]()

## 🌟 Features

- **High-Performance Face Detection**: YOLOv8x-based detection with CUDA acceleration
- **Comprehensive Attribute Analysis**: Age, gender, race, and emotion detection
- **Advanced Clustering**: Multiple clustering algorithms (DBSCAN, Agglomerative, K-means)
- **Rich Visualization**: 2D/3D embedding visualization using t-SNE and UMAP
- **Efficient Caching**: Smart caching system for embeddings and attributes
- **GPU Optimization**: CUDA support with CPU fallback
- **Attribute Analytics**: Detailed analysis of demographic and emotional patterns

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/DeepFace-Recognition-Clustering.git
cd DeepFace-Recognition-Clustering

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📋 Requirements

```plaintext
torch>=1.8.0
opencv-python>=4.5.0
deepface>=0.0.75
ultralytics>=8.0.0
numpy>=1.19.0
pandas>=1.2.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
umap-learn>=0.5.0
tqdm>=4.60.0
```

## 🚀 Quick Start

```python
from face_recognition_system import FaceRecognitionSystem

# Initialize the system
system = FaceRecognitionSystem(data_dir="path/to/your/images")

# Prepare and process the dataset
system.prepare_data()

# Visualize results
system.visualize_clusters_2d(method='tsne')
system.visualize_embeddings()

# Query similar faces
results = system.query_person("path/to/query/image.jpg")
```

## 📊 Dataset Requirements

- **Format**: JPG images
- **Recommended Resolution**: 68*128 pixels
- **Directory Structure**: Flat directory containing all images
- **Minimum Dataset Size**: No specific minimum, but works best with >1000 images
- **Testing Dataset Size**: System tested on 25,259 images

## 🎯 Use Cases

1. **Large-Scale Face Analysis**
   - Process and analyze large image datasets
   - Extract demographic and emotional patterns
   - Generate detailed attribute reports

2. **Face Clustering**
   - Automatic grouping of similar faces
   - Multiple clustering algorithms for different needs
   - Visualization of cluster distributions

3. **Similarity Search**
   - Find similar faces in large datasets
   - Query by image functionality
   - Attribute-based filtering

## 🔧 Advanced Configuration

```python
# Custom initialization with specific parameters
system = FaceRecognitionSystem(
    data_dir="path/to/images",
    batch_size=32,
    device="cuda",
    hf_token="your-huggingface-token"  # Optional
)

# Configure clustering parameters
system.cluster_faces(
    method='dbscan',
    eps=0.3,
    min_samples=2
)
```

## 📈 Performance Optimization

1. **GPU Memory Management**
   - Dynamic batch sizing
   - Automatic CPU fallback
   - Efficient cache management

2. **Processing Speed**
   - Multi-level caching system
   - Batch processing optimization
   - Parallel processing capabilities

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{deepface_recognition_clustering,
    title={DeepFace-Recognition-Clustering},
    author={Your Name},
    year={2024},
    url={https://github.com/yourusername/DeepFace-Recognition-Clustering}
}
```

## 📚 References

1. YOLOv8: Ultralytics, 2023
2. VGG-Face: Parkhi et al., 2015
3. DeepFace: Taigman et al., 2014
4. UMAP: McInnes et al., 2018


## 🙏 Acknowledgments

- DeepFace Framework developers
- Ultralytics team for YOLOv8
- The PyTorch community
- All contributors and users of this project