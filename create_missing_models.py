# create_missing_models.py
"""
Script pour créer les modèles manquants
"""

import torch
import torch.nn as nn
from pathlib import Path

class DemoModel(nn.Module):
    """Modèle de démonstration pour les tests"""
    def __init__(self, num_classes=28):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((10, 10))
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 10 * 10, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes * 6)  # 6 = x, y, w, h, conf, class
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        batch_size = x.size(0)
        return output.view(batch_size, -1, 6)

def create_models():
    """Crée les modèles de démonstration manquants"""
    models_dir = Path("storage/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    models_to_create = [
        ("best_extended_model.pth", "Modèle étendu 28 classes"),
        ("fast_stream_model.pth", "Modèle rapide pour streaming")
    ]
    
    for model_name, description in models_to_create:
        model_path = models_dir / model_name
        
        if not model_path.exists():
            print(f"🔧 Création de {model_name}...")
            
            # Créer le modèle
            model = DemoModel()
            
            # Sauvegarder avec métadonnées
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_info': {
                    'name': model_name,
                    'description': description,
                    'type': 'demo',
                    'classes': 28,
                    'input_size': (320, 320),
                    'version': '1.0.0'
                }
            }, model_path)
            
            print(f"✅ {model_name} créé ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")
        else:
            print(f"ℹ️  {model_name} existe déjà")
    
    print("\n🎉 Tous les modèles sont maintenant disponibles!")

if __name__ == "__main__":
    create_models()