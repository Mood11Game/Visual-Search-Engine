import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Собираем архитектуру сети
class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=512):
        super(EmbeddingNet, self).__init__()
        self.backbone = models.mobilenet_v3_small(weights=None)
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        return F.normalize(x, p=2, dim=1)

# Загружаем веса
model = EmbeddingNet(embedding_dim=512)
model.load_state_dict(torch.load('mobilenet_v3_embedding.pth', map_location='cpu'))
model.eval()

# Делаем экспорт
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, dummy_input, "visual_search.onnx",
    export_params=True, opset_version=14, do_constant_folding=True,
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print("Экспорт завершен! Файл visual_search.onnx создан.")
