import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Content type for inputs and outputs
JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'

# Model initialization with a pretrained ResNet50
def net():
    """
    Initialize the model using a pretrained ResNet50 model with custom final layer.
    """
    model = models.resnet50(pretrained=True)

    # Freeze all layers except the final layer
    for param in model.parameters():
        param.requires_grad = False   

    # Replace the final layer with a new layer for 5 classes (for bin item counts)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 5)
    )
    return model

# Model loading function
def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = net().to(device)  # Changed from Net() to net() for consistency
    model_path = os.path.join(model_dir, "model.pth")  # Ensure this matches the model file name

    # Load model parameters using state_dict if using a standard PyTorch save
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info("Model loaded successfully.")
    return model

# Input processing function
def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    logger.info("Processing input data.")
    
    # If input is an image in JPEG format
    if content_type == JPEG_CONTENT_TYPE:
        return Image.open(io.BytesIO(request_body))
    
    # If input is a JSON object containing an image URL
    elif content_type == JSON_CONTENT_TYPE:
        request = json.loads(request_body)
        url = request.get('url')
        if url:
            try:
                img_content = requests.get(url).content
                return Image.open(io.BytesIO(img_content))
            except requests.RequestException as e:
                logger.error(f"Error fetching image from URL: {e}")
                raise ValueError("Could not fetch image from URL.")
        else:
            raise ValueError("JSON request body must contain a 'url' field.")
    
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

# Prediction function
def predict_fn(input_object, model):
    logger.info("Making predictions.")
    
    # Define the image transformations
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Apply transformations and add a batch dimension
    input_tensor = test_transform(input_object).unsqueeze(0)
    
    # Move tensor to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_tensor)
    
    # Convert outputs to probabilities
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    predicted_class = probabilities.argmax().item()
    confidence = probabilities.max().item()
    
    logger.info(f"Predicted class: {predicted_class}, Confidence: {confidence:.4f}")
    
    return {
        "predicted_class": predicted_class,
        "confidence": confidence
    }

# Output serialization function
def output_fn(prediction, accept=JSON_CONTENT_TYPE):
    logger.info("Serializing output.")
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction), accept
    else:
        raise ValueError(f"Unsupported content type: {accept}")
