from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class EmitFrameInfo:
    offset_idx: int
    offset_s: float
    winsize_s: float
    mfcc_vec: np.ndarray # MFCC features [120]
    bilstm_h: np.ndarray   # biLSTM hidden vec [320]
    phone_logits: np.ndarray   # phone logits [230]
    phone_token: str    # phone token
    
    def __repr__(self):
        return "<EmitFrameInfo {phone} @{t:>6.2f}>".format(
          phone=self.phone_token,
          t=self.offset_s
        )

def build_emit_frame(
    winshift_s: float,
    winsize_s: float,
    emit_indices: np.ndarray,    
    mfcc_feats: np.ndarray, 
    lstm_outs: np.ndarray, 
    token_logits: np.ndarray, 
    token_phones: List[str]    
    ) -> List[EmitFrameInfo]:

    ret = []
    mfcc_feats = np.squeeze(mfcc_feats, axis=0)
    lstm_outs = np.squeeze(lstm_outs, axis=0)
    token_logits = np.squeeze(token_logits, axis=0)
    for emit_i, frame_idx in enumerate(emit_indices):
        frame_x = EmitFrameInfo(
            frame_idx,
            winshift_s * frame_idx,  # from lm/decoder.py:67
            winsize_s,
            mfcc_feats[frame_idx],
            lstm_outs[frame_idx],
            token_logits[frame_idx],
            token_phones[emit_i]
        )
        ret.append(frame_x)

    return ret