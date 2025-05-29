# config_stable.py - Configuration stabilisée pour éviter les NaN
config = {
    # === CLASSES ÉTENDUES AVEC PERSONNES ===
    'num_classes': 28,
    
    # === PARAMÈTRES STABILISÉS ===
    'batch_size': 2,  # RÉDUIT pour plus de stabilité
    'learning_rate': 0.0005,  # BEAUCOUP PLUS BAS pour éviter les NaN
    'num_epochs': 30,
    'image_size': (320, 320),  # PLUS PETIT pour stabilité
    'use_mixed_precision': False,  # DÉSACTIVÉ - cause souvent des NaN
    'max_train_images': 4000,  # RÉDUIT pour démarrage plus rapide
    'max_val_images': 400,
    'warmup_epochs': 5,  # PLUS LONG warmup
    
    # === OPTIMISATIONS STABILISÉES ===
    'optimizer': 'sgd',  # SGD plus stable qu'AdamW au début
    'weight_decay': 1e-5,  # RÉDUIT
    'momentum': 0.9,
    'num_workers': 0,  # DÉSACTIVÉ pour éviter les problèmes
    'pin_memory': False,  # DÉSACTIVÉ pour stabilité
    'gradient_clipping': 0.5,  # PLUS AGRESSIF pour éviter explosion
    
    # === PARAMÈTRES DE DÉTECTION CONSERVATEURS ===
    'confidence_threshold': 0.3,
    'nms_threshold': 0.5,
    'max_detections': 50,
    
    # === LOSS FUNCTIONS STABILISÉES ===
    'focal_loss': False,  # DÉSACTIVÉ temporairement
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    'iou_loss_weight': 1.0,  # RÉDUIT - était 2.0
    'class_loss_weight': 1.0,
    
    # === CHEMINS ===
    'coco_dir': 'c:/Users/ay855/Documents/detction_model/coco',
    'output_dir': 'output_stable_training',
    
    # === CLASSES (IDENTIQUES) ===
    'classes': [
        'person',       
        'backpack',      'suitcase',      'handbag',       'tie',
        'umbrella',      'hair drier',    'toothbrush',    'cell phone',
        'laptop',        'keyboard',      'mouse',         'remote',
        'tv',            'clock',         'microwave',     'bottle',
        'cup',           'bowl',          'knife',         'spoon',
        'fork',          'wine glass',    'refrigerator',  'scissors',
        'book',          'vase',          'chair',
    ],
    
    # === PONDÉRATION DÉSACTIVÉE TEMPORAIREMENT ===
    'class_weights': {},  # Vide pour éviter les complications
    
    # === NOMS EN FRANÇAIS ===
    'class_names_fr': {
        'person': 'Personne',
        'backpack': 'Sac à dos',
        'suitcase': 'Valise',
        'handbag': 'Sac à main',
        'tie': 'Cravate',
        'hair drier': 'Sèche-cheveux',
        'toothbrush': 'Brosse à dents',
        'cell phone': 'Téléphone',
        'laptop': 'Ordinateur portable',
        'keyboard': 'Clavier',
        'mouse': 'Souris',
        'remote': 'Télécommande',
        'tv': 'Télévision',
        'bottle': 'Bouteille',
        'cup': 'Tasse',
        'bowl': 'Bol',
        'knife': 'Couteau',
        'spoon': 'Cuillère',
        'fork': 'Fourchette',
        'wine glass': 'Verre',
        'scissors': 'Ciseaux',
        'book': 'Livre',
        'clock': 'Horloge',
        'umbrella': 'Parapluie',
        'vase': 'Vase',
        'chair': 'Chaise',
        'microwave': 'Micro-ondes',
        'refrigerator': 'Réfrigérateur'
    },
    
    # === PARAMÈTRES STABILISÉS ===
    'early_stopping_patience': 8,
    'validation_frequency': 3,  # Moins fréquent
    'ema_decay': 0.999,  # RÉDUIT pour plus de stabilité
    
    # === PARAMÈTRES POUR LA DÉTECTION DE PERSONNES ===
    'person_detection_params': {
        'min_person_size': 24,  # RÉDUIT
        'person_confidence_boost': 0.05,  # RÉDUIT
        'person_nms_threshold': 0.4,
    },
    
    # === DEBUGGING ===
    'debug_nan': True,  # Activer le debug des NaN
    'check_gradients': True,  # Vérifier les gradients
}