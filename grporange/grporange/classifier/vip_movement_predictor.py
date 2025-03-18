import torch
import torch.nn as nn

class VipMovementPredictor(nn.Module):
    def __init__(self, embed_dim=8, lstm_hidden=64, env_hidden=16, env_embed_dim=8, fc_hidden=32, num_moves=6):
        super(VipMovementPredictor, self).__init__()
        # Embedding pour les mouvements (6 catégories -> vecteurs de dimension embed_dim)
        self.move_embedding = nn.Embedding(num_embeddings=num_moves, embedding_dim=embed_dim)
        
        # LSTM pour la séquence des mouvements (entrée de dim embed_dim, sortie de dim lstm_hidden)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=lstm_hidden, batch_first=True)
        
        # Réseau pour l'environnement spatial
        self.env_fc1 = nn.Linear(9, env_hidden)        # 9 entrées -> env_hidden neurones
        self.env_fc2 = nn.Linear(env_hidden, env_embed_dim)  # env_hidden -> env_embed_dim neurones
        
        # Couche fully-connected pour combiner LSTM et environnement
        self.fc_comb = nn.Linear(lstm_hidden + env_embed_dim, fc_hidden)
        # Couche de sortie pour prédire la classe du prochain mouvement
        self.fc_out = nn.Linear(fc_hidden, num_moves)
        
    def forward(self, env_input, moves_seq):
        # env_input: tensor de shape (batch_size, 9)
        # moves_seq: tensor de shape (batch_size, seq_len=5)
        
        # 1. Traitement de la séquence des mouvements par l'LSTM
        # Embedding des mouvements
        embedded_seq = self.move_embedding(moves_seq)  
        # embedded_seq: shape (batch_size, 5, embed_dim)
        
        # Passage dans l'LSTM
        # lstm_out: sorties à chaque pas (batch_size, 5, lstm_hidden) - on n'en a pas forcément besoin ici
        # (h_n, c_n): états cachés et de cellule finaux de l'LSTM
        lstm_out, (h_n, c_n) = self.lstm(embedded_seq)
        # h_n: shape (num_layers, batch_size, lstm_hidden). On utilise le dernier état caché de la dernière couche:
        seq_features = h_n[-1]  # shape (batch_size, lstm_hidden)
        
        # 2. Traitement de l'environnement spatial
        env_features = torch.relu(self.env_fc1(env_input))     # première couche + activation ReLU
        env_features = torch.relu(self.env_fc2(env_features))  # seconde couche + ReLU
        # env_features: shape (batch_size, env_embed_dim)
        
        # 3. Combinaison des caractéristiques séquence + environnement
        combined = torch.cat([seq_features, env_features], dim=1)  # concaténation des vecteurs
        combined_features = torch.relu(self.fc_comb(combined))     # couche cachée combinée + ReLU
        
        # 4. Couche de sortie
        output = self.fc_out(combined_features)  # pas de Softmax ici, sera géré par la loss
        return output
