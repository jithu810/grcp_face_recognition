import os
import cv2
import numpy as np
import faiss
from insightface.app import FaceAnalysis
from collections import Counter
from utils.config import Config

project_root = Config.PROJECT_ROOT

loggers = Config.init_logging()
logger = loggers['chatservice']

class FaceRecognitionManager:
    def __init__(self, db_path=r"faces_db", index_path="saved_files/face_index.faiss", labels_path="saved_files/face_labels.npy", ctx_id=0):
        self.db_path = os.path.join(project_root, db_path)
        self.index_path = os.path.join(project_root, index_path)
        self.labels_path = os.path.join(project_root, labels_path)
        self.ctx_id = ctx_id
        self.embedding_dim = 512
        self.app = self._init_insightface()
        self.index, self.labels = self._load_index_and_labels()

    def _init_insightface(self):
<<<<<<< HEAD
        logger.info("Initializing InsightFace...")
        model_root = os.path.join(project_root, "models", "buffalo_l")
        app = FaceAnalysis(root=model_root)
        app.prepare(ctx_id=self.ctx_id)
=======
        print("🔧 Initializing InsightFace...")

        model_path = r"models\buffalo_l\models\buffalo_l\models"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        if not hasattr(self, 'ctx_id'):
            raise AttributeError("ctx_id is not defined in the class. Set self.ctx_id in __init__")

        try:
            app = FaceAnalysis(root=model_path)
            app.prepare(ctx_id=self.ctx_id)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize InsightFace: {e}")

>>>>>>> main
        return app

    def _load_index_and_labels(self):
        if os.path.exists(self.index_path) and os.path.exists(self.labels_path):
            logger.info("Loading existing FAISS index and labels...")
            index = faiss.read_index(self.index_path)
            labels = list(np.load(self.labels_path))
            logger.info(f"Loaded index with {index.ntotal} embeddings.")
        else:
            logger.warning("No existing index found. Creating a new one...")
            index = faiss.IndexFlatL2(self.embedding_dim)
            labels = []
        return index, labels

    def _save_index_and_labels(self):
        faiss.write_index(self.index, self.index_path)
        np.save(self.labels_path, np.array(self.labels))
        logger.info("Index and labels saved successfully.")

    def get_database_summary(self):
        """
        Returns a summary of the face database:
        - Total number of embeddings
        - A dictionary of labels and the count of embeddings for each
        """
        label_counts = Counter(self.labels)

        logger.info("Database Summary:")
        logger.info(f"Total Embeddings: {self.index.ntotal}")
        for label, count in label_counts.items():
            logger.info(f" - {label}: {count} embeddings")

        return {
            "total_embeddings": self.index.ntotal,
            "label_summary": dict(label_counts)
        }

    def _extract_embeddings(self, person_name):
        person_dir = os.path.join(self.db_path, person_name)
        embeddings = []
        labels = []

        if not os.path.isdir(person_dir):
            logger.error(f"Directory not found: {person_dir}")
            return embeddings, labels

        logger.info(f"Extracting embeddings for: {person_name}")
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Could not read: {img_path}")
                continue

            faces = self.app.get(img)
            if not faces:
                logger.warning(f"No face detected in: {img_name}")
                continue

            emb = faces[0].embedding
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
            labels.append(person_name)
            logger.info(f"Added embedding from: {img_name}")

        return embeddings, labels

    def add_new_person(self, person_name, similarity_threshold=0.4):
        logger.info(f"Adding new person: {person_name}")
        if person_name in self.labels:
            logger.warning(f"Person '{person_name}' already exists in database.")
            return

        new_embeddings, new_labels = self._extract_embeddings(person_name)
        if not new_embeddings:
            logger.warning("No valid embeddings found. Aborting.")
            return

        filtered_embeddings = []
        filtered_labels = []

        for emb in new_embeddings:
            emb = np.array([emb]).astype("float32")
            if self.index.ntotal > 0:
                D, I = self.index.search(emb, k=1)
                if D[0][0] < similarity_threshold:
                    logger.warning(f"Similar face found in DB (distance: {D[0][0]:.4f}). Skipping this image.")
                    continue
            filtered_embeddings.append(emb[0])
            filtered_labels.append(person_name)

        if not filtered_embeddings:
            logger.warning("All embeddings too similar to existing entries. Nothing added.")
            return

        self.index.add(np.array(filtered_embeddings).astype("float32"))
        self.labels.extend(filtered_labels)
        self._save_index_and_labels()

        logger.info(f"Added {len(filtered_embeddings)} unique embeddings for '{person_name}'.")

    def update_person(self, person_name):
        logger.info(f"Updating person: {person_name}")

        # Reconstruct existing index
        all_embeddings = self.index.reconstruct_n(0, self.index.ntotal)
        kept_embeddings = []
        kept_labels = []

        removed = 0
        for emb, label in zip(all_embeddings, self.labels):
            if label != person_name:
                kept_embeddings.append(emb)
                kept_labels.append(label)
            else:
                removed += 1

        logger.info(f"Removed {removed} old embeddings for '{person_name}'.")

        # Add new embeddings
        new_embeddings, new_labels = self._extract_embeddings(person_name)
        kept_embeddings.extend(new_embeddings)
        kept_labels.extend(new_labels)

        # Rebuild index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(np.array(kept_embeddings).astype("float32"))
        self.labels = kept_labels
        self._save_index_and_labels()

        logger.info(f"Person '{person_name}' updated with {len(new_embeddings)} new embeddings.")

    def delete_person(self, person_name):
        logger.info(f"Deleting person: {person_name}")

        if person_name not in self.labels:
            logger.warning(f"Person '{person_name}' not found in index.")
            return

        # Reconstruct all embeddings
        all_embeddings = self.index.reconstruct_n(0, self.index.ntotal)

        # Filter out embeddings of this person
        kept_embeddings = []
        kept_labels = []
        removed = 0

        for emb, label in zip(all_embeddings, self.labels):
            if label != person_name:
                kept_embeddings.append(emb)
                kept_labels.append(label)
            else:
                removed += 1

        if removed == 0:
            logger.warning(f"No embeddings found for: {person_name}")
            return

        logger.info(f"Removed {removed} embeddings for: {person_name}")

        # Rebuild FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        if kept_embeddings:
            self.index.add(np.array(kept_embeddings).astype("float32"))
        self.labels = kept_labels

        # Save changes
        self._save_index_and_labels()
        logger.info(f"Person '{person_name}' deleted successfully.")

    def verify_face(self, image_path, target_label, threshold=1):
        logger.info(f"Verifying face against ID: {target_label}")
        
        img = cv2.imread(image_path)
        if img is None:
            logger.error("Failed to read the image.")
            return False, None

        faces = self.app.get(img)
        if not faces:
            logger.warning("No face detected in the image.")
            return False, None

        test_embedding = faces[0].embedding
        test_embedding = test_embedding / np.linalg.norm(test_embedding)
        test_embedding = np.array([test_embedding]).astype("float32")

        D, I = self.index.search(test_embedding, 1)
        distance = D[0][0]
        matched_label = self.labels[I[0][0]]

        is_match = (str(matched_label) == str(target_label)) and (distance <= threshold)

        if not is_match:
            logger.warning(f"No match found for '{target_label}'. Closest match: '{matched_label}' "
                        f"with distance: {distance:.4f} (Threshold: {threshold})")
        else:
            logger.info(f"Match found — Target: {target_label}, Predicted: {matched_label}, Distance: {distance:.4f}")

        return bool(is_match), float(distance)

    def recognize_face(self, image_path, k=5, show=False):
        logger.info(f"\n Recognizing face from: {image_path}")
        
        img = cv2.imread(image_path)
        if img is None:
            logger.error(" Failed to read the image.")
            return "Invalid image", None

        faces = self.app.get(img)
        if not faces:
            logger.warning(" No face detected in the image.")
            return "No face detected", None

        test_face = faces[0]
        test_embedding = faces[0].embedding
        test_embedding = test_embedding / np.linalg.norm(test_embedding)
        test_embedding = np.array([test_embedding]).astype("float32")

        # Search top-k in FAISS
        D, I = self.index.search(test_embedding, k)

        logger.info("Top-K Match Results:")
        for rank, (dist, idx) in enumerate(zip(D[0], I[0]), start=1):
            matched_name = self.labels[idx]
            logger.info(f"  {rank}. {matched_name} (Distance: {dist:.4f})")

        top_k_labels = [self.labels[i] for i in I[0]]
        votes = Counter(top_k_labels)
        predicted_person, vote_count = votes.most_common(1)[0]
        confidence = 1.0 - D[0][0]  # Lower distance = higher confidence

        logger.info(f"\n Final Prediction: {predicted_person} (Votes: {vote_count}, Confidence: {confidence:.4f})")

        if show:
            bbox = test_face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"{predicted_person} ({confidence:.2f})"
            cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.imshow("Detected Face", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return predicted_person, confidence

if __name__ == "__main__":
    manager = FaceRecognitionManager()

    # manager.add_new_person("messi")
    # manager.update_person("ronaldo")
    # manager.delete_person("messi")

    image_path = r"D:\face-recognition\pred\m.jpg"
    target_label = "person1"  # replace with actual label you want to verify against

    # Step 1: Recognize face
    result, score = manager.recognize_face(image_path, k=3)
    print("\nPredicted Person:", result)
    print("Confidence Score:", round(score, 4) if score is not None else "N/A")

    # Step 2: Verify face
    is_match, distance = manager.verify_face(image_path, target_label)
    print(f"\nVerification against '{target_label}':", "Matched " if is_match else "Not Matched ")
    print("Distance Score:", round(distance, 4) if distance is not None else "N/A")
