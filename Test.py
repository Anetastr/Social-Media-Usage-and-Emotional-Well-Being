import numpy as np

features = np.random.rand(100, 4)  # 3 řádky, 4 sloupce
vektor_vah = np.random.rand(4)  # Vektor délky 10 s hodnotami z intervalu [0, 1)


def test_Matrix_Vector_Dimension_Compatibility():
    a = features.shape[1]
    b = vektor_vah.shape[0]

    expected = True
    result = np.isclose(a, b)
    assert result == expected