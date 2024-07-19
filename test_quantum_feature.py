import pandas as pd
from data_fetch import add_quantum_feature

# Create a simple DataFrame with a 'close' column
data = {
    'close': [100, 200, 300, 400, 500]
}
df = pd.DataFrame(data)

# Add quantum feature
df = add_quantum_feature(df)

print(df)
