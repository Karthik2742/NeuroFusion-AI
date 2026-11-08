"""
NeuroFusion AI - WebSocket Progress Manager
Real-time progress updates for long-running operations.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import WebSocket
from loguru import logger


class ProgressManager:
    """
    Manages WebSocket connections for real-time progress updates.
    
    Supports multiple concurrent operations with per-client tracking.
    """

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        self._lock = asyncio.Lock()

    async def connect(self, client_id: str, websocket: WebSocket) -> None:
        """Accept and register a WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self.active_connections[client_id] = websocket
        logger.info(f"WebSocket connected: {client_id}")

    async def disconnect(self, client_id: str) -> None:
        """Remove a WebSocket connection."""
        async with self._lock:
            self.active_connections.pop(client_id, None)
        logger.info(f"WebSocket disconnected: {client_id}")

    async def send_progress(
        self,
        client_id: str,
        operation: str,
        progress: float,
        message: str = "",
        data: dict[str, Any] | None = None,
    ) -> None:
        """
        Send a progress update to a specific client.
        
        Args:
            client_id: Client identifier
            operation: Operation name (e.g., 'enhance', 'preprocess')
            progress: Progress value [0.0, 1.0]
            message: Human-readable status message
            data: Optional additional data
        """
        ws = self.active_connections.get(client_id)
        if ws is None:
            return

        payload = {
            "type": "progress",
            "operation": operation,
            "progress": round(progress, 2),
            "message": message,
        }
        if data:
            payload["data"] = data

        try:
            await ws.send_text(json.dumps(payload))
        except Exception as e:
            logger.warning(f"Failed to send progress to {client_id}: {e}")
            await self.disconnect(client_id)

    async def send_complete(
        self,
        client_id: str,
        operation: str,
        result: dict[str, Any] | None = None,
    ) -> None:
        """Send operation completion notification."""
        ws = self.active_connections.get(client_id)
        if ws is None:
            return

        payload = {
            "type": "complete",
            "operation": operation,
            "progress": 1.0,
            "message": f"{operation} complete",
        }
        if result:
            payload["result"] = result

        try:
            await ws.send_text(json.dumps(payload))
        except Exception:
            pass

    async def send_error(
        self,
        client_id: str,
        operation: str,
        error: str,
    ) -> None:
        """Send error notification."""
        ws = self.active_connections.get(client_id)
        if ws is None:
            return

        payload = {
            "type": "error",
            "operation": operation,
            "message": error,
        }

        try:
            await ws.send_text(json.dumps(payload))
        except Exception:
            pass

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Broadcast a message to all connected clients."""
        disconnected = []
        for client_id, ws in self.active_connections.items():
            try:
                await ws.send_text(json.dumps(message))
            except Exception:
                disconnected.append(client_id)

        for cid in disconnected:
            await self.disconnect(cid)


# Global progress manager instance
progress_manager = ProgressManager()
