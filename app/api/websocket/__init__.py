"""
📡 WEBSOCKET PACKAGE - GESTION TEMPS RÉEL
========================================
Package pour la gestion des connexions WebSocket temps réel

Ce package gère:
- Connexions WebSocket multiples simultanées
- Communication bidirectionnelle avec les clients
- Streaming de détections en temps réel
- Notifications et événements
- Gestion des erreurs et déconnexions

Composants:
- WebSocketManager: Gestionnaire principal des connexions
- StreamHandler: Logique de traitement des streams
- Utilitaires de communication temps réel

Intégration:
- Next.js frontend pour interface webcam
- Spring Boot backend pour monitoring
- Service de détection IA pour résultats temps réel
"""

from .stream_handler import WebSocketManager, StreamHandler, ConnectionManager

__all__ = [
    "WebSocketManager",
    "StreamHandler", 
    "ConnectionManager"
]