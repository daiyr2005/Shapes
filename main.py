from fastapi import FastAPI, UploadFile, File,HTTPException
import io
import uvicorn
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

app = FastAPI()

class CheckImage(nn.Module):
  def __init__(self):
    super().__init__()
    self.first = nn.Sequential(
      nn.Conv2d(3, 16, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2),

      nn.Conv2d(16, 32, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2),

      nn.Conv2d(32, 64, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2),

      nn.Conv2d(64, 128, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2),

    )

    self.second = nn.Sequential(
        nn.Flatten(),
        nn.Linear(128 * 8* 8, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 4)
    )

  def forward(self, x):
    x = self.first(x)
    x= self.second(x)
    return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CheckImage()
model.load_state_dict(torch.load('model.pt', map_location=device))
model.to(device)
model.eval()

transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

classes = ['circle', 'square', 'star', 'triangle']
@app.post('/predict')
async def predict_image(image: UploadFile = File(...)):
    try:
        image_data = await image.read()

        if not image_data:
            raise HTTPException(status_code=400, detail="No image")
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_ten = transforms(img).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(image_ten)
            pred=y_pred.argmax(dim=1).item()


        return {
            "Answer": classes[pred]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8002)
