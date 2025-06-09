import torch
import torchvision.transforms as transforms
from PIL import Image


val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
  
])



def load_model(model_path: str):
    """
    Load a trained PyTorch model from disk (saved via torch.save(model, path))
    and set it to eval() mode.

    Args:
        model_path (str): Path to the .pth or .pt file containing your trained model.

    Returns:
        torch.nn.Module: The loaded model in eval mode (on CPU).
    """


    model = torch.load(
        model_path,
        map_location=torch.device("cpu"),
        weights_only=False,   # Allow loading the entire saved model object
    )
    model.eval()
    return model



def predict(model: torch.nn.Module, image: Image.Image) -> int:
    """
    Given a loaded model and a PIL.Image, return 0 (not drowsy) or 1 (drowsy).

    Args:
        model (torch.nn.Module): Your trained PyTorch model in eval() mode.
        image (PIL.Image.Image): A PIL image (RGB) of a human face.

    Returns:
        int: 0 if non-drowsy, 1 if drowsy.
    """
    # Apply the validation/test transform:
    image_tensor = val_test_transform(image)        # [3, 224, 224]
    image_tensor = image_tensor.unsqueeze(0)        # [1, 3, 224, 224]

    with torch.no_grad():
        outputs = model(image_tensor)               # assume shape [1, 2] or [1, 1]
        # If your model outputs two logits (for classes 0 vs 1):
        if outputs.dim() == 2 and outputs.shape[1] == 2:
            # e.g. softmax‐based two‐class output
            _, predicted = torch.max(outputs, dim=1)
            return int(predicted.item())
        else:
            # If your model outputs a single logit (e.g. using `nn.Linear(…) -> [1, 1]`):
            # apply a sigmoid threshold of 0.5
            prob = torch.sigmoid(outputs).item()
            return 1 if prob >= 0.5 else 0
