class ModelConfig:
    # Model architecture
    BACKBONE: str = "efficientnet-b0"
    FPN_CHANNELS: int = 256
    NUM_CLASSES: int = 80
    
    # Training settings
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.001
    WEIGHT_DECAY: float = 0.0001
    
    # Inference settings
    CONFIDENCE_THRESHOLD: float = 0.5
    NMS_THRESHOLD: float = 0.5
    MAX_DETECTIONS: int = 100

model_config = ModelConfig() 