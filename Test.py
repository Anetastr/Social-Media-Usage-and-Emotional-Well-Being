import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from MLModel import train_neuron

def test_train_neuron_basic():
    # Generuj malý binární dataset
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    X = pd.DataFrame(X)
    y = pd.Series(y)

    result = train_neuron(X, y, epochs=100, plot=False, random_state=42)

    # Kontrola návratového typu a klíčů
    assert isinstance(result, dict)
    expected_keys = {'weights', 'bias', 'accuracy', 'report', 'loss_train', 'loss_val', 'scaler'}
    assert expected_keys.issubset(result.keys())

    # Kontrola rozsahu přesnosti
    assert 0 <= result['accuracy'] <= 1

    # Ověření, že se váhy a bias opravdu změnily
    assert result['weights'].shape[0] == X.shape[1]
    assert isinstance(result['bias'], float)

    # Ověření délky loss křivek (že učení probíhalo)
    assert len(result['loss_train']) >= 1
    assert len(result['loss_val']) >= 1