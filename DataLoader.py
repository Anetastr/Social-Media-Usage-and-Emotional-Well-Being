import csv

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.header = []
        self.data = []

        try:
            self._load_data()
        except FileNotFoundError:
            print(f"soubor nenalezen: {file_path}")
        except IsADirectoryError:
            print(f"Očekáván soubor, ale '{file_path}' je složka.")

    def _load_data(self):
        """Načtení CSV souboru"""
        with open(self.file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)

            try:
                self.header = next(reader)
            except StopIteration:
                raise ValueError("Soubor je prázdný!")

            self.data = [row for row in reader]

    def shape(self):
        """Vrací tvar dat (počet řádků, počet sloupců)"""
        rows = len(self.data)
        cols = len(self.header)
        return (rows, cols)

    def columns(self):
        """Vrací názvy sloupců"""
        return self.header

    def head(self, n=5):
        """Vrací prvních n řádků"""
        return self.data[:n]

# Cesta k datum
file_path = "data/train.csv"

# Objekt
loader = DataLoader(file_path)

# Výstupy
print("Shape:", loader.shape())
print("Názvy sloupců:", loader.columns())
print("Prvních 5 řádků:")
for row in loader.head():
    print(row)