from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, validator
from enum import Enum

import shutil
import uuid
import os
import json
import base64
import matplotlib
matplotlib.use('Agg')

from PIL import Image
import numpy as np
import torch
import faiss

from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException

from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from deepface import DeepFace


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

text_model = SentenceTransformer('paraphrase-MiniLM-L12-v2')
yolo_model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
face_model = YOLO(yolo_model_path)

attribute_weights = {
    "name": 0.5,
    "national_id": 0.3,
    "governorate": 0.1,
    "city": 0.05,
    "street": 0.05
}

THRESHOLD = 0.35



def translate_text_Card(text, target_lang="en"):
    try:
        if not text or len(text) < 3:
            return text
        detected_lang = detect(text)
        if detected_lang != target_lang:
            return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except LangDetectException:
        return text
    return text

def calculate_text_similarity_Card(lost: dict, found: dict):
    total_score = 0
    total_weight = sum(attribute_weights.values())
    for attr, weight in attribute_weights.items():
        text1 = translate_text_Card(str(getattr(lost, attr, "")))
        text2 = translate_text_Card(str(getattr(found, attr, "")))
        emb1 = text_model.encode(text1, convert_to_tensor=True)
        emb2 = text_model.encode(text2, convert_to_tensor=True)
        sim = util.pytorch_cos_sim(emb1, emb2).item()
        total_score += sim * weight
    return total_score / total_weight

def detect_and_crop_face_Card(image_path, prefix="face"):
    img = Image.open(image_path).convert("RGB")
    result = face_model(img, classes=[0])
    boxes = result[0].boxes.xyxy.cpu().numpy()
    scores = result[0].boxes.conf.cpu().numpy()
    for i, score in enumerate(scores):
        if score > 0.5:
            x1, y1, x2, y2 = map(int, boxes[i])
            face = img.crop((x1, y1, x2, y2))
            output_path = f"static/faces/{prefix}_face.jpg"
            os.makedirs("static/faces", exist_ok=True)
            face.save(output_path)
            return output_path
    return None


async def save_to_file_system_Card(prefix, image_dir, json_path, json_key,
                                name, national_id, governorate, city, street, contact, image, image_name):
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
        
    
    face_filename = image_name 
    face_path = os.path.join(image_dir, face_filename)
    image.file.seek(0)
    with open(face_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    if image:
        temp_path = f"temp_{uuid.uuid4().hex}.jpg"
        image.file.seek(0)
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        cropped = detect_and_crop_face_Card(temp_path, prefix=face_filename.replace(".jpg",""))
        if cropped:
            shutil.copy(cropped, face_path)
        else:
            shutil.move(temp_path, face_path)
        os.remove(temp_path)

    # تحميل بيانات JSON
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {json_key: []}

    data[json_key].append({
        "name": name,
        "national_id": national_id,
        "governorate": governorate,
        "city": city,
        "street": street,
        "contact": contact,
        "image_url": f"http://localhost:8000/{face_path.replace(os.sep, '/')}"
    })

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return {"status": "added", "image": f"http://localhost:8000/{face_path.replace(os.sep, '/')}"}

@app.post("/add_lost_card")
async def add_lost(
    name: str = Form(...),
    national_id: str = Form(...),
    governorate: str = Form(...),
    city: str = Form(...),
    street: str = Form(...),
    contact: str = Form(...),
    image_name: str = Form(...),
    image: UploadFile = File(None)
):
    return await save_to_file_system_Card(
        prefix="losted",
        image_dir="static/lostedcard",
        json_path="metadata/lostedcard/lostedcard.json",
        json_key="losted",
        name=name,
        national_id=national_id,
        governorate=governorate,
        city=city,
        street=street,
        contact=contact,
        image=image,
        image_name=image_name
    )
@app.post("/add_found_card")
async def add_found(
    name: str = Form(...),
    national_id: str = Form(...),
    governorate: str = Form(...),
    city: str = Form(...),
    street: str = Form(...),
    contact: str = Form(...),
    image_name: str = Form(...),
    image: UploadFile = File(None)
):
    return await save_to_file_system_Card(
        prefix="founded",
        image_dir="static/foundedcard",
        json_path="metadata/foundedcard/foundedcard.json",
        json_key="founded",
        name=name,
        national_id=national_id,
        governorate=governorate,
        city=city,
        street=street,
        contact=contact,
        image=image,
        image_name=image_name
    )
class MatchType(str, Enum):
    text = "text"
    image = "image"
    both = "both"

class Lost_card(BaseModel):
    name: str
    national_id: str
    governorate: str
    city: str
    street: str
    contact: str
    image_name: str

class MatchRequest_card(BaseModel):
    match_type: MatchType
    lost: Lost_card
    
@app.post("/match_card/")
async def match(request: MatchRequest_card):
    match_type = request.match_type
    lost = request.lost

    metadata_path = "metadata/foundedcard/foundedcard.json"
    if not os.path.exists(metadata_path):
        return {"error": "foundedcard.json not found"}

    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    found_list = data.get("founded", [])
    lost_dict = lost  

    best_score = -1
    text_best_match = None
    image_best_match = None
    text_score = None
    face_verified = None
    face_distance = None
    lost_face_url = None
    found_face_url = None

    if match_type in ["text", "both"]:
        for found_data in found_list:
            score = calculate_text_similarity_Card(lost_dict, found_data)
            if score > best_score:
                best_score = score
                text_best_match = found_data
        text_score = best_score

    if match_type in ["image", "both"]:
        image_name = lost.image_name
        lost_img_path = os.path.join("static/lostedcard", image_name)
        found_images_dir = "static/foundedcard/"
        found_images = os.listdir(found_images_dir)


        for found_img_name in found_images:
            found_img_path = os.path.join(found_images_dir, found_img_name)
            try:
                result = DeepFace.verify(
                    img1_path=lost_img_path,
                    img2_path=found_img_path,
                    model_name="Facenet512",
                    distance_metric="euclidean_l2",
                    threshold=0.3
                )
                if result["verified"]:
                    face_verified = True
                    face_distance = result["distance"]
                    lost_face_url = f"http://localhost:8000/static/lostedcard/{image_name}"
                    found_face_url = f"http://localhost:8000/static/foundedcard/{found_img_name}"
                    image_best_match = next((item for item in found_list if item.get("image_url", "").endswith(found_img_name)), None)
                    break
            except Exception as e:
                print(f"Face matching error: {e}")
                face_verified = False
                face_distance = None

    if match_type == "text":
        best_match = text_best_match
        final_result = text_score is not None and text_score > THRESHOLD
    elif match_type == "image":
        best_match = image_best_match
        final_result = face_verified
    else:  # both
        best_match = text_best_match  # أو ممكن تختار حسب المنطق الذي تريده
        final_result = (text_score is not None and text_score > THRESHOLD) and face_verified

    return {
        "text_similarity": round(text_score, 4) if text_score is not None else None,
        "face_verified": face_verified,
        "face_distance": face_distance,
        "match_result": final_result,
        "face_images": {
            "lost_face": f"{lost_face_url}" if lost_face_url else None,
            "found_face": f"{found_face_url}" if found_face_url else None
        },
        "contact_info": {
            "found": best_match.get("contact") if best_match else None
        }
}



clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
text_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


attributes = {
    "brand": 0.3,
    "color": 0.2,
    "governorate": 0.2,
    "city": 0.15,
    "street": 0.15,
}

def translate_text_phone(text, target_lang="en"):
    if not text or len(text) < 3:
        return text
    try:
        detected_lang = detect(text)
        if detected_lang != target_lang:
            return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except LangDetectException:
        return text
    return text

def calculate_similarity_phone(lost, found):
    total_score = 0
    total_weight = 0  

    for attr, weight in attributes.items():
        if attr in lost and attr in found:
            text1 = translate_text_phone(str(getattr(lost, attr, "")))
            text2 = translate_text_phone(str(getattr(found, attr, "")))
            embedding1 = text_model.encode(text1, convert_to_tensor=True)
            embedding2 = text_model.encode(text2, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
            total_score += similarity * weight
            total_weight += weight

    if total_weight == 0:
        return 0 

    return total_score / total_weight



def get_image_embedding_phone(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    return image_features[0].cpu().numpy()


def build_faiss_index_phone(image_folder):
    embeddings = []
    paths = []
    for img in os.listdir(image_folder):
        if img.startswith("."):
            continue
        path = os.path.join(image_folder, img)
        emb = get_image_embedding_phone(path)
        embeddings.append(emb)
        paths.append(path)
    embedding_matrix = np.vstack(embeddings).astype("float32")
    faiss.normalize_L2(embedding_matrix)
    index = faiss.IndexFlatIP(embedding_matrix.shape[1])
    index.add(embedding_matrix)
    return index, paths


def find_most_similar_images_faiss_phone(input_image_path, image_folder, top_k=3):
    emb = get_image_embedding_phone(input_image_path).reshape(1, -1).astype("float32")
    faiss.normalize_L2(emb)
    index, paths = build_faiss_index_phone(image_folder)
    distances, indices = index.search(emb, top_k)
    results = []
    for i in range(top_k):
        result = {
            "image_path": paths[indices[0][i]],
            "similarity": float(distances[0][i])
        }
        results.append(result)
    return results

async def save_to_file_system_phone(prefix, image_dir, json_path, json_key,
                                governorate, city, street, contact,
                            brand, color, image, image_name):
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    image_filename = image_name
    image_path = os.path.join(image_dir, image_filename)

    image.file.seek(0)
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {json_key: []}

    data[json_key].append({
        "governorate": governorate,
        "city": city,
        "street": street,
        "contact": contact,
        "brand": brand,
        "color": color,
        "image_url": f"http://localhost:8000/{image_path.replace(os.sep, '/')}"
    })

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return {"status": "added", "image": f"http://localhost:8000/{image_path.replace(os.sep, '/')}"}

@app.post("/add_found_phone")
async def add_found(
    governorate: str = Form(...),
    city: str = Form(...),
    street: str = Form(...),
    contact: str = Form(...),
    image_name: str = Form(...),
    brand: str = Form(...),
    color: str = Form(...),
    image: UploadFile = File(None)
):
    return await save_to_file_system_phone(
        prefix="founded",
        image_dir="static/foundedphone",
        json_path="metadata/foundedphone/foundedphone.json",
        json_key="founded",
        governorate=governorate,
        city=city,
        street=street,
        contact=contact,
        brand=brand,
        color=color,
        image=image,
        image_name=image_name
    )


    
class Lost_phone(BaseModel):
    governorate: str
    city: str
    street: str
    contact: str
    brand: str
    color: str
    image_name: str
    image: str  

class MatchRequest_phone(BaseModel):
    match_type: MatchType
    lost: Lost_phone




@app.post("/match_phone/")
async def match(request: MatchRequest_phone):
    match_type = request.match_type
    lost = request.lost
    lost_dict = lost.dict()
    image = lost.image

    metadata_path = "metadata/foundedphone/foundedphone.json"
    if not os.path.exists(metadata_path):
        return JSONResponse(status_code=404, content={"error": "foundedphone.json not found"})

    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    found_list = data.get("founded", [])
    if not found_list:
        return JSONResponse(status_code=404, content={"error": "No found items available"})

    text_score = 0
    image_score = 0
    final_score = 0
    text_best_match = None
    matched_images = []

    temp_path = None  
    try:
        if match_type in ["image", "both"] and image:
            temp_filename = lost.image_name
            UPLOAD_FOLDER = "static/uploads"
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
            with open(temp_path, "wb") as f:
                    f.write(base64.b64decode(image))
                
            IMAGE_FOLDER = "static/foundedphone"
            image_results = find_most_similar_images_faiss_phone(temp_path, IMAGE_FOLDER, top_k=3)
            image_score = image_results[0]["similarity"] if image_results else 0

            for res in image_results:
                image_name = os.path.basename(res["image_path"])
                associated_data = next(
                    (item for item in found_list if os.path.basename(item.get("image_url", "")) == image_name),
                    None
                )
                matched_images.append({
                    "image_url": f"http://localhost:8000/static/foundedphone/{image_name}",
                    "image_similarity": round(res["similarity"], 4),
                    "associated_data": associated_data
                })

        if match_type in ["text", "both"]:
            best_score = -1
            for found_data in found_list:
                score = calculate_similarity_phone(found_data, lost_dict)
                if score > best_score:
                    best_score = score
                    text_best_match = found_data
            text_score = best_score

        if match_type == "text":
            final_score = text_score
        elif match_type == "image":
            final_score = image_score
        elif match_type == "both":
            final_score = (text_score * 0.5) + (image_score * 0.5)

        matched = 1 if final_score >= 0.7 else 0

        return JSONResponse({
            "match_type": match_type,
            "matched": matched,
            "final_score": round(final_score, 4),

            "text_similarity": round(text_score, 4),
            "text_best_match": text_best_match,

            "image_similarity": round(image_score, 4),
            "matched_images": matched_images
        })

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
