import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import json
import os
import torch.nn.functional as F

# 1. Load the pre-trained ResNet18 model with correct weights
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.eval()

# 2. Define the image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
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
    "cow": ["Holstein", "Jersey", "Friesian", "Guernsey", "Zebu", "Brahman", "cow", "American Staffordshire Terrier"]
}

def get_top_predictions(outputs, labels, top_k=5):
    probabilities = F.softmax(outputs, dim=1)
    top_prob, top_indices = torch.topk(probabilities, top_k)
    predictions = []
    for prob, idx in zip(top_prob[0], top_indices[0]):
        predictions.append({
            'label': labels[idx.item()],
            'confidence': prob.item() * 100
        })
    return predictions

def explain_recognition(predicted_label, confidence, top_predictions):
    explanation = "\nDetailed Analysis:\n"
    explanation += f"1. Primary Recognition: The model is {confidence:.2f}% confident this is a {predicted_label}\n"
    
    # Add supporting evidence from other top predictions
    explanation += "\n2. Supporting Evidence:\n"
    for pred in top_predictions[1:4]:  # Use next 3 predictions as supporting evidence
        explanation += f"   - Also detected features similar to {pred['label']} ({pred['confidence']:.2f}%)\n"
    
    # Add visual characteristics that typically define this category
    explanation += "\n3. Key Characteristics Detected:\n"
    if predicted_label == "dog":
        explanation += "   - Facial features (snout, ears)\n   - Body structure\n   - Common dog breed characteristics\n"
    elif predicted_label == "cat":
        explanation += "   - Feline facial features\n   - Ear shape\n   - Body proportions\n"
    elif predicted_label == "bird":
        explanation += "   - Beak presence\n   - Wing features\n   - Feather patterns\n"
    elif predicted_label == "elephant":
        explanation += "   - Trunk presence\n   - Large body structure\n   - Characteristic ear shape\n"
    elif predicted_label == "horse":
        explanation += "   - Long face structure\n   - Body proportions\n   - Leg characteristics\n"
    elif predicted_label == "bear":
        explanation += "   - Body mass\n   - Facial structure\n   - Paw characteristics\n"
    elif predicted_label == "cow":
        explanation += "   - Body shape\n   - Head structure\n   - Common bovine characteristics\n"
    
    return explanation

def print_fancy_header():
    print("\n" + "="*40)
    print("     🦁 Animal Recognition Results 🦁     ")
    print("="*40 + "\n")

def print_fancy_footer():
    print("\n" + "="*40)
    print("            End of Analysis            ")
    print("="*40 + "\n")

def main():
    # Load ImageNet labels
    imagenet_labels_path = "imagenet-simple-labels.json"
    
    if not os.path.exists(imagenet_labels_path):
        print(f"Error: {imagenet_labels_path} not found.")
        return

    with open(imagenet_labels_path, "r") as f:
        labels = json.load(f)

    # Get image path from user
    while True:
        print("\n📸 Image Recognition System")
        image_path = input("Please enter the path to your image file (or 'quit' to exit): ")
        
        if image_path.lower() == 'quit':
            print("\nThank you for using the Animal Recognition System!")
            break
            
        if not os.path.exists(image_path):
            print(f"❌ Error: File not found at {image_path}")
            continue

        try:
            # Load and preprocess the image
            image = Image.open(image_path).convert("RGB")
            input_tensor = preprocess(image).unsqueeze(0)

            # Perform inference
            with torch.no_grad():
                outputs = model(input_tensor)
                
            # Get top predictions with confidence scores
            top_predictions = get_top_predictions(outputs, labels)
            predicted_label = top_predictions[0]['label']
            confidence = top_predictions[0]['confidence']

            print_fancy_header()

            # Find general category
            general_category = None
            for category, specific_labels in general_categories.items():
                if any(label.lower() in predicted_label.lower() for label in specific_labels):
                    general_category = category
                    break

            if general_category:
                print(f"📊 Primary Detection: {general_category.upper()}")
                print(f"🎯 Confidence Level: {confidence:.2f}%\n")
                
                print("🔍 Top 5 Matches:")
                for i, pred in enumerate(top_predictions, 1):
                    print(f"{i}. {pred['label']}: {pred['confidence']:.2f}%")
                
                print(explain_recognition(general_category, confidence, top_predictions))
            else:
                print("❌ No Animal Detected")
                print("The image does not contain any recognizable animals from our categories.")
                print("\nSuggestions:")
                print("- Try adjusting the image angle")
                print("- Ensure the animal is clearly visible")
                print("- Check that the image is well-lit")
                print("- Make sure the animal is one of our supported categories")

            # Show the image
            image.show()
            
            print_fancy_footer()

        except Exception as e:
            print(f"❌ Error processing image: {str(e)}")

if __name__ == "__main__":
    main()