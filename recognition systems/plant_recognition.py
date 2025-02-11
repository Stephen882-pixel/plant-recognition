import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import json
import os

# Load the model
print("Loading ResNet18 model...")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.eval()

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Expanded plant categories mapping
plant_categories = {
    "flower": [
        "daisy", "rose", "sunflower", "orchid", "lily", "iris",
        "tulip", "lotus", "magnolia", "chrysanthemum", "rapeseed",
        "cardoon", "hippeastrum", "water lily", "morning glory",
        "garden phlox", "anemone", "dahlia", "hibiscus", "marigold",
        "black-eyed susan", "bellflower", "poppy", "geranium"
    ],
    "tree": [
        "pine", "oak", "maple", "palm", "fig", "banyan",
        "deciduous tree", "conifer", "evergreen", "juniper",
        "eucalyptus", "cypress", "spruce", "birch", "willow",
        "japanese maple", "beech", "cherry", "apple tree"
    ],
    "vegetable": [
        "cabbage", "broccoli", "lettuce", "corn", "cucumber",
        "bell pepper", "carrot", "potato plant", "artichoke",
        "cauliflower", "spinach", "kale", "eggplant", "asparagus",
        "celery", "leek", "zucchini", "radish"
    ],
    "fruit": [
        "apple", "orange", "banana", "grape", "strawberry",
        "tomato", "pineapple", "lemon", "peach", "pear",
        "blueberry", "raspberry", "mango", "pomegranate",
        "fig", "plum", "kiwi", "watermelon"
    ],
    "succulent": [
        "cactus", "aloe", "jade plant", "echeveria", "agave",
        "haworthia", "sedum", "sempervivum", "zebra plant",
        "string of pearls", "burro's tail"
    ],
    "herb": [
        "mint", "basil", "rosemary", "lavender", "sage",
        "thyme", "parsley", "cilantro", "oregano", "chives",
        "dill", "fennel", "chamomile", "tarragon"
    ],
    "crop": [
        "rapeseed", "wheat", "corn", "barley", "soybean",
        "oat", "rice", "sunflower", "cotton", "sugarcane",
        "quinoa", "flax", "canola"
    ],
    "wild_plant": [
        "fern", "moss", "lichen", "wildflower", "weed",
        "thistle", "nettle", "clover", "dandelion", "ivy",
        "bramble", "bracken", "heather"
    ]
}

def analyze_plant_image():
    """Analyze plant image with user interaction"""
    # Check for ImageNet labels file
    if not os.path.exists("imagenet-simple-labels.json"):
        print("\nWarning: ImageNet labels file not found!")
        print("Please ensure imagenet-simple-labels.json is in the same directory")
        return

    print("\nCurrent working directory:", os.getcwd())
    print("\nEnter the path to your image file:")
    image_path = input()
    
    print(f"\nAttempting to analyze: {image_path}")
    
    try:
        # Load labels
        with open("imagenet-simple-labels.json", "r") as f:
            labels = json.load(f)
        
        # Verify image exists
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            return
            
        # Load and preprocess image
        print("Loading and preprocessing image...")
        image = Image.open(image_path).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0)
        
        # Get model predictions with top-3 results
        print("Running inference...")
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top3_prob, top3_idx = torch.topk(probabilities, 3)
            
        # Check top-3 predictions for plant categories
        found_category = None
        found_label = None
        highest_confidence = 0
        
        print("\nTop 3 predictions:")
        for prob, idx in zip(top3_prob, top3_idx):
            label = labels[idx.item()].lower()
            confidence = prob.item() * 100
            print(f"- {label}: {confidence:.2f}%")
            
            # Check each prediction against plant categories
            for category, specific_labels in plant_categories.items():
                if any(plant_label.lower() in label for plant_label in specific_labels):
                    if confidence > highest_confidence:
                        found_category = category
                        found_label = label
                        highest_confidence = confidence
        
        print("\n=== Plant Recognition Results ===")
        if found_category:
            print(f"General Category: {found_category.title()}")
            print(f"Specific Type: {found_label.title()}")
            print(f"Confidence: {highest_confidence:.2f}%")
        else:
            print("No plant detected in the image or unable to categorize.")
            print("Try adjusting the image or using a different angle.")
        print("=============================\n")
        
        # Show image
        image.show()
            
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    analyze_plant_image()