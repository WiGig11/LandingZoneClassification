import torch
import torch.onnx
import torch.nn as nn

# 定义模型结构
class Model(nn.Module):
    def __init__(self, num_classes=3):  # 确保 num_classes 与训练时一致
        super(Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 实例化模型并加载状态字典
model = Model(num_classes=2)  # 根据您的模型类别数调整
model.load_state_dict(torch.load('models/letter_detector.pth'))
model.eval()
model = model.to('cpu')

# 准备输入张量
input_tensor = torch.randn(1, 1, 28, 28).to('cpu')  # 确保输入张量在 CPU 上

# 导出模型为 ONNX 格式
torch.onnx.export(
    model,
    input_tensor,
    "letter_detector.onnx",
    export_params=True,
    opset_version=10,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'}, 
        'output': {0: 'batch_size'}
    }
)
