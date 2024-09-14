import torch
from PIL import Image
from torchvision import transforms
from AMQI import AMQI  # AMQI


def predict():
    test_img_name = ["2071_mos1.42.jpg", "1564_mos2.58.jpg", "1073_mos4.42.jpg"]
    test_img_mos = [1.42, 2.58, 4.42]
    device = torch.device((f"cuda:0" if torch.cuda.is_available() else "cpu"))
    model = AMQI()
    weight = torch.load("./model/AMQI_epoch_10.pth", map_location="cpu")
    model.load_state_dict(weight)
    del weight
    model.to(device)
    model.eval()

    with torch.no_grad():
        trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        for i in range(len(test_img_name)):
            img = trans(Image.open(f"./input/{test_img_name[i]}")).unsqueeze(0).to(device)
            score = model(img)
            print(f"Test img: {test_img_name[i]}, predict score: {round(score.item(), 2)}, mos: {test_img_mos[i]}")


def predict2(test_img_path=None):
    device = torch.device((f"cuda:0" if torch.cuda.is_available() else "cpu"))
    model = AMQI()
    weight = torch.load("./model/AMQI_epoch_10.pth", map_location="cpu")
    model.load_state_dict(weight)
    del weight
    model.to(device)
    model.eval()

    with torch.no_grad():
        trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        img = trans(Image.open(test_img_path)).unsqueeze(0).to(device)
        score = model(img)
        print(f"Predict score: {round(score.item(), 2)}")


if __name__ == '__main__':
    pass
    # 1
    predict()
    # Test img: 2071_mos1.42.jpg, predict score: 1.85, mos: 1.42
    # Test img: 1564_mos2.58.jpg, predict score: 2.27, mos: 2.58
    # Test img: 1073_mos4.42.jpg, predict score: 4.2, mos: 4.42
    # 2
    predict2(test_img_path="./input/1564_mos2.58.jpg")
    # Predict score: 2.27
