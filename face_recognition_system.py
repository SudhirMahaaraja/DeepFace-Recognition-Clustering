import os
import logging
from pathlib import Path
import torch
import numpy as np
import cv2
from deepface import DeepFace
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from huggingface_hub import login
from ultralytics import YOLO
from tqdm import tqdm
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

try:
    import umap.umap_ as umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logging.warning("UMAP not available. Install with 'pip install umap-learn'")


class PersonAttributes:
    def __init__(self, device=device):
        self.analyzer = DeepFace
        self.device = device

    def extract_attributes(self, image_path: str) -> Dict:
        try:
            analysis = self.analyzer.analyze(
                img_path=str(image_path),
                actions=['age', 'gender', 'race', 'emotion'],
                enforce_detection=False
            )
            if isinstance(analysis, list):
                analysis = analysis[0]

            img = cv2.imread(str(image_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pixels = img.reshape(-1, 3)
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=3)
            kmeans.fit(pixels)
            dominant_colors = kmeans.cluster_centers_.astype(int)

            return {
                'age': analysis['age'],
                'gender': analysis['gender'],
                'dominant_race': analysis['dominant_race'],
                'emotion': analysis['dominant_emotion'],
                'dominant_colors': dominant_colors.tolist()
            }
        except Exception as e:
            logging.warning(f"Failed to extract attributes from {image_path}: {str(e)}")
            return {}


class FaceDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None, device=device):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.device = device
        self.attributes = {}
        self._extract_attributes()

    def _extract_attributes(self):
        attribute_extractor = PersonAttributes(device=self.device)
        for path in tqdm(self.image_paths, desc="Extracting attributes"):
            self.attributes[path] = attribute_extractor.extract_attributes(path)

    def __len__(self):
        return len(self.image_paths)

    def preprocess_image(self, image_path):
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        return img

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = self.preprocess_image(img_path)

        if self.transform:
            img = self.transform(img)
            img = img.to(self.device)

        item = {'image': img, 'path': img_path, 'attributes': self.attributes[img_path]}
        if self.labels is not None:
            item['label'] = torch.tensor(self.labels[idx], device=self.device)
        return item


class FaceRecognitionSystem:
    def __init__(self, data_dir="E:\CVintern\sample_dataset", batch_size=32, device=device, hf_token=None):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.device = device
        self.hf_token = hf_token
        self.embeddings_cache = {}
        self.clusters = None
        self.embedding_model = None

        if self.hf_token:
            login(self.hf_token)

        # Initialize YOLO model with CUDA if available
        self.face_detector = YOLO("yolov8x.pt")
        if torch.cuda.is_available():
            self.face_detector.to(self.device)

        self.logger = logging.getLogger(__name__)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def detect_faces(self, image_path: str) -> np.ndarray:
        """
        Detect faces in an image using YOLO model.
        Includes fallback for CUDA compatibility issues.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            self.logger.warning(f"Could not read image: {image_path}")
            return np.array([])

        try:
            # First try with CUDA
            results = self.face_detector.predict(source=img, imgsz=640, device=self.device)
            bboxes = results[0].boxes.xyxy.cpu().numpy()
        except NotImplementedError:
            self.logger.warning("CUDA NMS not available, falling back to CPU for detection")
            # Move model to CPU temporarily for this prediction
            self.face_detector.cpu()
            results = self.face_detector.predict(source=img, imgsz=640, device='cpu')
            bboxes = results[0].boxes.xyxy.numpy()
            # Move model back to GPU for other operations
            if torch.cuda.is_available():
                self.face_detector.to(self.device)
        except Exception as e:
            self.logger.error(f"Face detection failed for {image_path}: {str(e)}")
            return np.array([])

        return bboxes

    def extract_embedding(self, img_path: str) -> Optional[np.ndarray]:
        if img_path in self.embeddings_cache:
            return self.embeddings_cache[img_path]

        try:
            result = DeepFace.represent(
                str(img_path),
                model_name="VGG-Face",
                enforce_detection=False
            )
            if isinstance(result, list) and len(result) > 0:
                embedding = np.array(result[0]['embedding'])
            else:
                embedding = np.array(result['embedding'])

            embedding_tensor = torch.from_numpy(embedding).to(self.device)
            self.embeddings_cache[img_path] = embedding_tensor.cpu().numpy()
            return self.embeddings_cache[img_path]
        except Exception as e:
            self.logger.warning(f"Failed to process {img_path}: {str(e)}")
            return None

    def cluster_faces(self, method='dbscan'):
        """Cluster face embeddings using different methods."""
        if method == 'dbscan':
            clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
        elif method == 'agglomerative':
            from sklearn.cluster import AgglomerativeClustering
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.4,
                metric='cosine',
                linkage='average'
            )
        elif method == 'kmeans':
            from sklearn.cluster import KMeans
            # Estimate number of clusters
            from sklearn.metrics import silhouette_score
            best_k = 2  # default
            best_score = -1
            for k in range(2, min(10, len(self.embeddings))):
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(self.embeddings)
                score = silhouette_score(self.embeddings, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            clustering = KMeans(n_clusters=best_k, random_state=42)

        labels = clustering.fit_predict(self.embeddings)
        return labels

    def prepare_data(self):
        """Prepare and cluster the dataset."""
        image_paths = list(self.data_dir.glob("*.jpg"))
        print(f"Total images found: {len(image_paths)}")

        image_paths = list(self.data_dir.glob("*.jpg"))
        if not image_paths:
            raise ValueError(f"No images found in {self.data_dir}")

        embeddings = []
        valid_paths = []

        for img_path in tqdm(image_paths, desc="Processing images"):
            bboxes = self.detect_faces(img_path)
            if len(bboxes) > 0:
                embedding = self.extract_embedding(img_path)
                if embedding is not None:
                    embeddings.append(embedding)
                    valid_paths.append(img_path)

        embeddings = np.array(embeddings)
        # Verify normalization
        norms = np.linalg.norm(embeddings, axis=1)
        print("Embedding norms before normalization:", norms[:5])

        embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]

        # Verify after normalization
        norms_after = np.linalg.norm(embeddings, axis=1)
        print("Embedding norms after normalization:", norms_after[:5])

        # Add this debug code before clustering
        print("Embedding shape:", embeddings.shape)
        print("Sample embeddings distances:")
        from sklearn.metrics.pairwise import cosine_distances
        distances = cosine_distances(embeddings[:5])
        print(distances)

        # More flexible clustering parameters
        clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
        labels = clustering.fit_predict(embeddings)

        # Add this after clustering to see distribution
        unique_labels = set(labels)
        print(f"Number of clusters found: {len(unique_labels)}")
        print(f"Cluster sizes: {[list(labels).count(i) for i in unique_labels]}")

        self.clusters = {
            'embeddings': embeddings,
            'labels': labels,
            'paths': valid_paths
        }

        self.logger.info(
            f"\n[Cluster Report] 🧐\nUnique Persons Found: {len(set(labels))}\nProcessed Images: {len(valid_paths)}\n")
        self.dataset = FaceDataset([str(x) for x in valid_paths], labels, self.transform)

    def query_person(self, query_image_path: str, top_k: int = 5) -> List[Dict]:
        """Retrieve similar faces given a query image."""
        query_embedding = self.extract_embedding(query_image_path)
        if query_embedding is None:
            return []

        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Calculate similarities
        similarities = np.dot(self.clusters['embeddings'], query_embedding)
        most_similar_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in most_similar_indices:
            results.append({
                'path': str(self.clusters['paths'][idx]),
                'similarity': similarities[idx],
                'cluster': self.clusters['labels'][idx],
                'attributes': self.dataset.attributes[str(self.clusters['paths'][idx])]
            })

        return results

    def visualize_embeddings(self, method: str = 'tsne', plot_file: str = 'embedding_visualization.png'):
        """Visualize the distribution of face embeddings in 2D/3D."""
        embeddings = self.clusters['embeddings']
        labels = self.clusters['labels']

        # Standardize the embeddings
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(embeddings)

        if method.lower() == 'tsne':
            reducer = TSNE(n_components=3, random_state=42)
        else:  # UMAP
            reducer = umap.UMAP(n_components=3, random_state=42)

        embedded = reducer.fit_transform(scaled_embeddings)

        # Create 3D visualization
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(embedded[:, 0], embedded[:, 1], embedded[:, 2],
                             c=labels, cmap='tab20', alpha=0.6)
        plt.colorbar(scatter)

        ax.set_title(f'Face Embeddings Visualization ({method.upper()})')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')

        plt.savefig(plot_file)
        plt.close()

        # Create attribute visualization
        self._visualize_attributes(plot_file.replace('.png', '_attributes.png'))

    def _visualize_attributes(self, plot_file: str):
        """Visualize the distribution of attributes across clusters."""
        # Prepare data for visualization
        processed_data = []
        for path, attrs in self.dataset.attributes.items():
            cluster = self.clusters['labels'][self.clusters['paths'].index(Path(path))]
            if isinstance(attrs, dict):  # Check if attrs is a dictionary
                row = {
                    'cluster': cluster,
                    'gender': attrs.get('gender', 'Unknown'),
                    'age': float(attrs.get('age', 0)),
                    'dominant_race': attrs.get('dominant_race', 'Unknown'),
                    'emotion': attrs.get('dominant_emotion', 'Unknown')
                }
                processed_data.append(row)

        # Convert to DataFrame
        df = pd.DataFrame(processed_data)

        # Convert categorical columns
        df['cluster'] = df['cluster'].astype('category')
        df['gender'] = df['gender'].astype('category')
        df['dominant_race'] = df['dominant_race'].astype('category')
        df['emotion'] = df['emotion'].astype('category')

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        # Gender distribution
        try:
            sns.countplot(data=df, x='cluster', hue='gender', ax=axes[0, 0])
            axes[0, 0].set_title('Gender Distribution by Cluster')
        except Exception as e:
            logging.warning(f"Failed to plot gender distribution: {e}")
            axes[0, 0].text(0.5, 0.5, 'Gender plot unavailable', ha='center')

        # Age distribution
        try:
            sns.boxplot(data=df, x='cluster', y='age', ax=axes[0, 1])
            axes[0, 1].set_title('Age Distribution by Cluster')
        except Exception as e:
            logging.warning(f"Failed to plot age distribution: {e}")
            axes[0, 1].text(0.5, 0.5, 'Age plot unavailable', ha='center')

        # Dominant race distribution
        try:
            sns.countplot(data=df, x='cluster', hue='dominant_race', ax=axes[1, 0])
            axes[1, 0].set_title('Race Distribution by Cluster')
        except Exception as e:
            logging.warning(f"Failed to plot race distribution: {e}")
            axes[1, 0].text(0.5, 0.5, 'Race plot unavailable', ha='center')

        # Emotion distribution
        try:
            sns.countplot(data=df, x='cluster', hue='emotion', ax=axes[1, 1])
            axes[1, 1].set_title('Emotion Distribution by Cluster')
        except Exception as e:
            logging.warning(f"Failed to plot emotion distribution: {e}")
            axes[1, 1].text(0.5, 0.5, 'Emotion plot unavailable', ha='center')

        plt.tight_layout()
        plt.savefig(plot_file)
        plt.close()

    def visualize_clusters_2d(self, method='tsne'):
        """Visualize clusters in 2D with a simpler approach."""
        embeddings = self.clusters['embeddings']
        labels = self.clusters['labels']

        # Reduce dimensionality to 2D
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        else:  # UMAP
            if not UMAP_AVAILABLE:
                logging.warning("UMAP not available, falling back to t-SNE")
                reducer = TSNE(n_components=2, random_state=42)
                method = 'tsne'
            else:
                reducer = umap.UMAP(n_components=2, random_state=42)

        embedded = reducer.fit_transform(embeddings)

        # Create scatter plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embedded[:, 0], embedded[:, 1],
                              c=labels, cmap='tab20', alpha=0.6)
        plt.colorbar(scatter)
        plt.title(f'Face Embeddings Visualization ({method.upper()})')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')

        # Save plot
        plt.savefig(f'clusters_2d_{method}.png')
        plt.close()


if __name__ == "__main__":
    # Log CUDA information
    if torch.cuda.is_available():
        logging.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"Number of CUDA devices: {torch.cuda.device_count()}")
        logging.info(f"Current CUDA device: {torch.cuda.current_device()}")
    else:
        logging.warning("CUDA is not available. Using CPU.")

    system = FaceRecognitionSystem(data_dir="E:\CVintern\sample_dataset")
    system.prepare_data()

    system.visualize_clusters_2d(method='tsne')
    if UMAP_AVAILABLE:
        system.visualize_clusters_2d(method='umap')

    query_image = "E:\CVintern\quer.jpg"  # Replace with actual query image path
    results = system.query_person(query_image)

    for result in results:
        print(f"Match found: {result['path']}")
        print(f"Similarity: {result['similarity']:.3f}")
        print(f"Cluster: {result['cluster']}")
        print(f"Attributes: {result['attributes']}\n")
