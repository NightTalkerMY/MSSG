import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalGraphNet(nn.Module):
    """
    A pure PyTorch implementation of the Causal MSSG Graph.
    Avoids torch_geometric dependencies for easier Raspberry Pi 5 deployment.
    """
    def __init__(self, in_features=128, out_features=128):
        super(CausalGraphNet, self).__init__()
        # The Graph Convolution weight matrix
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, x, mask):
        # x shape: [Batch, Speakers(5), Frames(50), 128]
        # mask shape: [Batch, Speakers(5), Frames(50)]
        B, S, T, D = x.shape
        
        # Flatten spatial and temporal dimensions to create nodes: [B, S*T, 128]
        x_flat = x.view(B, S * T, D)
        mask_flat = mask.view(B, S * T)
        
        # 1. Initialize an empty Adjacency Matrix A [B, S*T, S*T]
        # A will dictate which nodes can "talk" to which other nodes.
        A = torch.zeros((B, S * T, S * T), device=x.device)
        
        # 2. Build the Causal Edges (Intra-speaker and Inter-speaker)
        for s in range(S):
            for t in range(T):
                current_node = s * T + t
                
                # A. Intra-speaker (Temporal): Connect to own past frames
                # Strict Causal Rule: t_prev must be <= t
                for t_prev in range(max(0, t - 3), t + 1): # Look back up to 3 frames
                    past_node = s * T + t_prev
                    A[:, current_node, past_node] = 1.0
                    
                # B. Inter-speaker (Spatial): Connect to other speakers in the EXACT SAME frame
                for other_s in range(S):
                    if other_s != s:
                        other_node = other_s * T + t
                        A[:, current_node, other_node] = 1.0

        # 3. Apply the Ghost Hunter Mask!
        # If a node is a ghost (mask == 0), sever all connections to it
        valid_matrix = mask_flat.unsqueeze(1) * mask_flat.unsqueeze(2) # [B, S*T, S*T]
        A = A * valid_matrix
        
        # Normalize the adjacency matrix to prevent exploding gradients
        row_sum = A.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        A_norm = A / row_sum
        
        # 4. Perform the Graph Convolution: Output = A_norm * (X * W)
        node_transform = torch.matmul(x_flat, self.weight) # [B, S*T, 128]
        graph_out = torch.bmm(A_norm, node_transform)      # [B, S*T, 128]
        
        # Reshape back to our 4D structure and add a residual connection
        graph_out = graph_out.view(B, S, T, D)
        return F.relu(graph_out + x)

class CausalStudentASD(nn.Module):
    NUM_SCALES = 7
    SCALE_DIM = 128  # Each of the 7 visual scales is 128-dim in the cache

    def __init__(self, hidden_dim=128, num_classes=2):
        super(CausalStudentASD, self).__init__()

        # 1. Per-scale visual projections (preserves scale independence)
        # Each mimics one of the teacher's 7 encoder pipelines
        self.scale_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.SCALE_DIM, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
            for _ in range(self.NUM_SCALES)
        ])

        # 2. Spatial-aware scale attention
        # Learns which visual scales matter based on face position/size [cx,cy,w,h]
        # Mimics teacher's face-size masking: large face → face/face_large/face_small/face_down
        #                                     small face → face/face_body/face_large/background/face_body_large
        self.scale_attn = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, self.NUM_SCALES),
        )

        # 3. Scale fusion → hidden_dim
        self.fuse = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
        )

        # 4. Causal Transformer Encoder (2 layers, higher dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=256,
            dropout=0.2, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 6. The Causal Graph
        self.causal_graph = CausalGraphNet(in_features=hidden_dim, out_features=hidden_dim)

        # 7. The Final Classifier (with dropout)
        self.feat_dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, v, a, s_cues, mask):
        """
        v: Visual features [B, 5, 50, 896] — 7 scales × 128-dim concatenated
        a: Audio features [B, 5, 50, 128]
        s_cues: Spatial info [B, 5, 50, 4] — face position/size [cx, cy, w, h]
        mask: Valid node mask [B, 5, 50]
        """
        B, S, T, _ = v.shape

        # Step 1: Per-scale visual processing (each scale projected independently)
        scale_feats = []
        for i, proj in enumerate(self.scale_projs):
            scale_v = v[:, :, :, i * self.SCALE_DIM:(i + 1) * self.SCALE_DIM]  # [B,S,T,128]
            scale_feats.append(proj(scale_v))  # [B,S,T,64]
        scale_stack = torch.stack(scale_feats, dim=3)  # [B,S,T,7,64]

        # Step 2: Spatial-aware scale attention
        # Attention weights decide which visual scales matter for this face
        attn_weights = F.softmax(self.scale_attn(s_cues), dim=-1)  # [B,S,T,7]
        v_fused = (scale_stack * attn_weights.unsqueeze(-1)).sum(dim=3)  # [B,S,T,64]

        # Step 3: Fuse to hidden_dim + audio
        x = self.fuse(v_fused) + a  # [B,S,T,128]

        # Step 4: Causal Transformer (each speaker independently)
        D = x.shape[-1]
        x_seq = x.view(B * S, T, D)
        causal_mask = torch.triu(torch.ones(T, T, device=v.device) * float('-inf'), diagonal=1)
        seq_features = self.transformer(x_seq, mask=causal_mask)  # [B*S, T, 128]

        # Reshape back to the scene matrix
        seq_features = seq_features.view(B, S, T, D)

        # Zero out ghost speaker outputs
        seq_features = seq_features * mask.unsqueeze(-1)

        # Step 6: Inter/Intra Speaker Graph Routing
        graph_features = self.causal_graph(seq_features, mask)

        # Step 7: Classification
        logits = self.classifier(self.feat_dropout(graph_features))  # [B, S, T, 2]

        # Return logits (for CE) and pre-graph features (for KD MSE)
        return logits, seq_features