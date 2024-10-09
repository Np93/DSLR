import pandas as pd
from sklearn.metrics import accuracy_score

def evaluate_predictions(predictions_path, true_data_path):
    # Charger les données de prédiction
    pred_df = pd.read_csv(predictions_path)
    
    # Charger les données réelles
    true_df = pd.read_csv(true_data_path)
    
    # Fusionner les prédictions avec les vraies étiquettes en utilisant l'index comme clé
    # Cela garantit que chaque prédiction est comparée à la bonne étiquette
    merged_df = pd.merge(true_df, pred_df, on="Index", suffixes=('_true', '_pred'))
    
    # Calculer la précision
    accuracy = accuracy_score(merged_df['Hogwarts House_true'], merged_df['Hogwarts House_pred'])
    print(f"Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    evaluate_predictions('houses.csv', 'data/test_data.csv')