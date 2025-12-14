import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder

projectDir = Path(__file__).parent.parent
dataPath = projectDir / "data" / "crimes.csv"
plotsPath = projectDir / "plots"

# load dataset
data = pd.read_csv(str(dataPath))

# -------------------------------------------------
# Normalization of numeric features
# -------------------------------------------------
numeric_feats = [
    "hour_float",
    "latitude",
    "longitude",
    "victim_age",
    "temp_c",
    "humidity",
    "dist_precinct_km",
    "pop_density",
]
# print(data[numeric_feats].head(5))


scaler = MinMaxScaler()
data[numeric_feats] = scaler.fit_transform(data[numeric_feats])

# keep only 3 decimal digits max for each numeric feature
for i in numeric_feats:
    data[i] = data[i].round(3)

# print(data[numeric_feats].head(5))

# -------------------------------------------------
# Normalization of categorical features
# -------------------------------------------------
categorical_feats = [
    "weapon_code",
    "scene_type",
    "weather",
]
# print(data[categorical_feats].head(5))

# Creating the encoder
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

encoded_array = encoder.fit_transform(data[categorical_feats])

encoded_df = pd.DataFrame(
    encoded_array, 
    columns=encoder.get_feature_names_out(categorical_feats)
)

# join new columns 
data = pd.concat([data, encoded_df],axis=1)
# save to .csv
data.to_csv( projectDir / "data" /'data_encoded.csv', index=False)