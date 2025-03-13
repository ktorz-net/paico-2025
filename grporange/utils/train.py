import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from classifier import VipMovementPredictor

# 1. Chargement des données
df = pd.read_csv("./data-map1-history5.csv")  # Remplacer par le vrai chemin du fichier CSV

# 2. Encodage des mouvements (derniers mouvements et mouvement cible)
move_mapping = {-1: 0,  0: 1,  3: 2,  6: 3,  9: 4,  12: 5}
move_cols = [f"move_{i}" for i in range(5)]  # ['move_0', ..., 'move_4']
for col in move_cols + ["next_move"]:
    df[col] = df[col].map(move_mapping)

# 3. Encodage de l'environnement spatial (mur = 1, libre = 0)
env_cols = [f"tile_{i}" for i in range(9)]  # ['tile_0', ..., 'tile_8']
# Remplacer -1 par 1, conserver 0
df[env_cols] = df[env_cols].replace({-1: 1, 0: 0})

# 4. Séparation des caractéristiques (X) et de la cible (y)
X_env = df[env_cols].values            # Environnement, shape (n_samples, 9)
X_moves = df[move_cols].values         # Séquence des 5 derniers mouvements, shape (n_samples, 5)
y = df["next_move"].values            # Mouvement suivant (étiquette)

# 5. Division train/test (80% train, 20% test)
X_env_train, X_env_test, X_moves_train, X_moves_test, y_train, y_test = train_test_split(
    X_env, X_moves, y, test_size=0.2, random_state=42
)

# 6. Conversion en tenseurs PyTorch
X_env_train = torch.tensor(X_env_train, dtype=torch.float32)
X_env_test  = torch.tensor(X_env_test, dtype=torch.float32)
X_moves_train = torch.tensor(X_moves_train, dtype=torch.long)
X_moves_test  = torch.tensor(X_moves_test, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test  = torch.tensor(y_test, dtype=torch.long)

# Vérification des formes des tenseurs
print("Entraînement - Environnement:", X_env_train.shape,
      "| Séquence mouvements:", X_moves_train.shape,
      "| Étiquettes:", y_train.shape)
print("Test - Environnement:", X_env_test.shape,
      "| Séquence mouvements:", X_moves_test.shape,
      "| Étiquettes:", y_test.shape)

# Création du Dataset d'entraînement et de test
train_dataset = TensorDataset(X_env_train, X_moves_train, y_train)
test_dataset  = TensorDataset(X_env_test, X_moves_test, y_test)

# Création des DataLoaders pour itérer par batch
batch_size = 64  # on peut ajuster la taille de batch
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialisation du modèle
model = VipMovementPredictor()
print(model)

# Sélection du dispositif d'exécution (GPU si disponible sinon CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositif d'exécution :", device)
model.to(device)  # envoie le modèle sur le GPU (ou reste sur CPU si pas de GPU)

# Définition de la fonction de perte et de l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Hyperparamètres d'entraînement
num_epochs = 50  # par exemple, on peut ajuster selon la convergence

# Boucle d'entraînement
for epoch in range(1, num_epochs+1):
    model.train()  # mode entraînement
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for env_batch, moves_batch, labels_batch in train_loader:
        # 1. Chargement du batch sur le bon dispositif (GPU ou CPU)
        env_batch = env_batch.to(device)
        moves_batch = moves_batch.to(device)
        labels_batch = labels_batch.to(device)
        
        # 2. Réinitialisation des gradients
        optimizer.zero_grad()
        
        # 3. Forward pass : prédire le prochain mouvement
        outputs = model(env_batch, moves_batch)  # outputs shape: (batch_size, 6)
        
        # 4. Calcul de la loss
        loss = criterion(outputs, labels_batch)
        
        # 5. Backpropagation
        loss.backward()
        
        # 6. Mise à jour des poids
        optimizer.step()
        
        # 7. Accumuler la perte pour affichage
        running_loss += loss.item() * env_batch.size(0)
        # 8. (Optionnel) Calcul de la précision sur ce batch pour suivi
        _, predicted = torch.max(outputs, 1)  # classe prédite
        correct_train += (predicted == labels_batch).sum().item()
        total_train += labels_batch.size(0)
    
    # Calcul des métriques de l'époque
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct_train / total_train
    
    # Affichage des résultats pour l'époque
    print(f"Époque {epoch}/{num_epochs} - Perte entraînement: {epoch_loss:.4f} - Précision entraînement: {epoch_acc:.4f}")

model.eval()  # mode évaluation
correct = 0
total = 0

with torch.no_grad():  # désactive la calcul des gradients
    for env_batch, moves_batch, labels_batch in test_loader:
        env_batch = env_batch.to(device)
        moves_batch = moves_batch.to(device)
        labels_batch = labels_batch.to(device)
        
        outputs = model(env_batch, moves_batch)
        # Prédiction la plus probable
        _, predicted = torch.max(outputs, 1)
        total += labels_batch.size(0)
        correct += (predicted == labels_batch).sum().item()

test_accuracy = correct / total
print(f"Précision sur le jeu de test: {test_accuracy:.4f}")

torch.save(model.state_dict(), "./classifier/weights/vip_movement_predictor.pth")

# model = MonModele()  # créer l'architecture du modèle
# model.load_state_dict(torch.load("model_weights.pth"))
# model.eval()  # mode 

# # Exemple d'utilisation sur une nouvelle situation (env_new, moves_new)
# model.eval()
# env_new = torch.tensor([[0,0,0, -1,0,-1, 0,0,0]], dtype=torch.float32).to(device)  # environnement 3x3 (à adapter)
# moves_new = torch.tensor([[2, 3, 3, 4, 1]], dtype=torch.long).to(device)  # derniers mouvements encodés (exemple)
# with torch.no_grad():
#     output = model(env_new, moves_new)
#     predicted_class = torch.argmax(output, dim=1).item()
# # Conversion inverse de l'encodage pour obtenir le mouvement prédict en valeur originale
# inv_mapping = {v: k for k, v in move_mapping.items()}
# predicted_move = inv_mapping[predicted_class]
# print("Mouvement prédit:", predicted_move)