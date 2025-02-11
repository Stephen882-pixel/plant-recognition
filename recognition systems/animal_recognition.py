import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import json
import os

# 1. Load the pre-trained ResNet18 model with correct weights
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # Ensure to use correct weights for the model
model.eval()  # Set the model to evaluation mode

# 2. Define the image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to 224x224 (required by ResNet18)
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Normalization for ImageNet models
        std=[0.229, 0.224, 0.225]
    )
])

# 3. Define a mapping of specific ImageNet labels to general categories
general_categories = {
    "dog": [
        "golden retriever", "german shepherd", "beagle", "dalmatian", "pomeranian",
        "chihuahua", "bulldog", "poodle", "labrador retriever", "Pembroke Welsh Corgi", 
        "English foxhound",
    ],
    "cat": [
        "tabby", "Siamese cat", "Persian cat", "Egyptian cat", "tiger cat"
    ],
    "elephant": ["African elephant", "Indian elephant"],
    "horse": ["Arabian horse", "Clydesdale"],
    "bird": ["peacock", "parrot", "ostrich", "eagle"],
    "bear": ["polar bear", "brown bear", "panda"],
    "cow": ["Holstein", "Jersey", "Friesian", "Guernsey", "Zebu", "Brahman", "cow","American Staffordshire Terrier"]  # New cow category added
}

# 4. Load ImageNet labels
imagenet_labels_path = "imagenet-simple-labels.json"

# Ensure the file exists at the specified path
if not os.path.exists(imagenet_labels_path):
    print(f"Error: {imagenet_labels_path} not found.")
else:
    with open(imagenet_labels_path, "r") as f:
        labels = json.load(f)

    # 5. Load and preprocess the input image
    image_path = "images/dog.jpeg"  # Replace with your image path
    image = Image.open(image_path).convert("RGB")  # Ensure image is RGB
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

    # 6. Perform inference
    with torch.no_grad():
        outputs = model(input_tensor)  # Get model predictions
        _, predicted_idx = outputs.max(1)  # Get the index of the highest probability

    # 7. Map the predicted index to a class label
    predicted_label = labels[predicted_idx.item()]
    print(f"Predicted label: {predicted_label}")  # Print the predicted label for debugging

    # 8. Generalize the predicted label with flexible matching (allows partial matches)
    general_category = None
    for category, specific_labels in general_categories.items():
        if any(label.lower() in predicted_label.lower() for label in specific_labels):  # Case insensitive match
            general_category = category
            break

    # 9. Output the result
    if general_category:
        print(f"The image contains a: {general_category}")
    else:
        print("The image is not recognizable as an animal.")

    # 10. Optional: Show the image (for verification)
    image.show()
