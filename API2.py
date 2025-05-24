import requests
import io
import pandas as pd

API_KEY = "de882d30"  # <-- Zadej svůj API klíč
SCHEMA_NAME = "690246"  # <-- např. "my_social_media_schema"

url = f"https://api.mockaroo.com/api/my_scheme.json?key=de882d30&count=100"

try:
    response = requests.get(url)
    response.raise_for_status()

    df = pd.read_json(io.StringIO(response.text))
    print(df.head())

    df.to_json("mockaroo_data.csv", index=False)
    print("Data byla uložena do souboru 'mockaroo_data.json'")

except requests.exceptions.RequestException as e:
    print("Chyba při stahování dat:", e)
