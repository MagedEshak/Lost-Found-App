import importlib

modules = [
    # Core
    "fastapi", "uvicorn", "numpy", "torch", "jinja2",

    # Vision & AI
    "ultralytics", "deepface", "cv2", "PIL", "matplotlib",
    "sentence_transformers", "transformers", "faiss",
    
    # Language & Utils
    "huggingface_hub", "deep_translator", "langdetect", "geopy",

    # AI Model Dependencies
    "tensorflow", "keras", "scipy", "sklearn", "pandas", "yaml",
    "google.protobuf", "gdown", "tqdm", "seaborn", "psutil",

    # Face Detection
    "mtcnn", "retinaface",

    # Optimization
    "tokenizers", "safetensors",

    # Extra Keras-related
    "keras_applications", "keras_preprocessing", "tf_keras",
]

print("\n🧪 Starting import test...\n")

for module in modules:
    try:
        importlib.import_module(module)
        print(f"✅ {module}")
    except Exception as e:
        print(f"❌ {module} —> {e}")

print("\n✅✅ Done.\n")
