import cv2
import numpy as np
import torch
import os
from facenet_pytorch import InceptionResnetV1
from chromadb.utils.data_loaders import ImageLoader
import chromadb
from ultralytics import YOLO
from PIL import Image

# Load YOLO model
model = YOLO("yolov8n.pt")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

sample_paths = []
for dir, dir_name, files in os.walk("/home/greatness-within/Downloads/animals/new samples"):
    for file_name in files:
        path = os.path.join(dir, file_name)
        sample_paths.append(path)

def main(pathy):
    img = cv2.imread(pathy)
    def get_boxes(img):
        results = model(img)  # Return a list of Results objects
        for result in results:
            boxes = result.boxes.xyxy[0]
            return boxes.cpu().detach().tolist()


    def crop(img, xyxy):
        color_converted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(color_converted)
        im = img_pil.crop((xyxy[0], xyxy[1], xyxy[2], xyxy[3]))
        return im


    def embs(im):
        numpy_image = np.array(im)
        im = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        face_roi = cv2.resize(im, (160, 160))
        face_roi = np.array(face_roi) / 255.0
        face_roi = torch.Tensor(face_roi).permute(2, 0, 1).unsqueeze(0).to(device)

        # Extract face embedding
        embedding = resnet(face_roi)
        list_embeddings = [embedding.detach().cpu().numpy()[0].tolist()]
        return list_embeddings[0]


    # Initialize ChromaDB client and collection
    client = chromadb.PersistentClient(path="/home/greatness-within/Downloads/animal_embs")
    data_loader = ImageLoader()
    collection = client.get_or_create_collection(name="animal_embs_new_10", data_loader=data_loader,
                                                 metadata={"hnsw:space": "cosine"})

    # Get bounding box
    bbox = get_boxes(img)
    crops = crop(img, bbox)

    # Get embedding and query collection
    embedding = embs(crops)
    results = collection.query(query_embeddings=embedding)
    predicted_name = results["metadatas"][0][0]["animals"] if len(results["metadatas"]) > 0 else "Unknown"

    # Draw bounding box and predicted name
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Adjust text position to make sure it's visible
    text_position = (x1, y1 - 20 if y1 - 10 > 10 else y1 + 10)
    cv2.putText(img, predicted_name, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the image
    cv2.imshow("Animal Detection", img)

    # Wait until a key is pressed and then close the window
    if cv2.waitKey(0) & 0xFF == 27:  # 27 is the Escape key
        cv2.destroyAllWindows()

prob_samples = []
for path in sample_paths:
    try:
        main(path)
    except:
        prob_samples.append(path)
        continue
print(prob_samples)