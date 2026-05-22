import torch
import cv2
import numpy as np
from typing import Dict, Tuple

class ViTExplainer:
    ''' generates the attention rollout heatmaps for the vit '''
    def __init__(self, model, discard_ratio: float = 0.9, device: str = 'cpu'):
        self.model = model
        self.discard_ratio = discard_ratio
        self.device = torch.device(device)
        self.attentions = []
        self._orig_forwards = []

    def _patch_blocks(self): 
        self._orig_forwards.clear()
        self.attentions.clear()

        for block in self.model.blocks:
            attn = block.attn
            self._orig_forwards.append(attn.forward)

            def make_patched(m, store):
                def patched(x, **kwargs):
                    B, N, C = x.shape
                    hd = C // m.num_heads 
                    qkv = m.qkv(x).reshape(B, N, 3, m.num_heads, hd).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv.unbind(0)
                    dots = (q @ k.transpose(-2, -1)) * m.scale
                    dots = dots.softmax(dim=-1)
                    store.append(dots.detach().cpu()) # Store the attention weights
                    out = (m.attn_drop(dots) @ v).transpose(1, 2).reshape(B, N, C)
                    return m.proj_drop(m.proj(out))
                return patched
            
            attn.forward = make_patched(attn, self.attentions)

    def _restore_blocks(self):
        for block, orig in zip(self.model.blocks, self._orig_forwards):
            block.attn.forward = orig
        self._orig_forwards.clear()

    def _rollout(self) -> np.ndarray:
        result = None
        for attn in self.attentions:
            a = attn[0]
            a = a + torch.eye(a.shape[-1])
            a = a / a.sum(dim=-1, keepdim=True)
            a = a.mean(dim=0)
            a[a < torch.quantile(a.view(-1), self.discard_ratio)] = 0
            
            result = a if result is None else torch.matmul(a, result)
            
        mask = result[0, 1:]
        g = int(mask.shape[0] ** 0.5)
        mask = mask.reshape(g, g).numpy()
        
        # Normalize between 0 and 1
        return (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    
    def generate_heatmap(self, face_tensor: torch.Tensor, original_crop: np.ndarray) -> Dict[str, np.ndarray]:
        self._patch_blocks()
        
        try:
            with torch.no_grad():
                _ = self.model(face_tensor)
        finally:
            self._restore_blocks()
            
        mask = self._rollout()
        
        h, w = original_crop.shape[:2]
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_CUBIC)
        
        mask_uint8 = np.uint8(255 * mask_resized)
        heatmap_color = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(original_crop, 0.5, heatmap_color, 0.5, 0)
        
        return {
            "heatmap": heatmap_color,
            "overlay": overlay
        }