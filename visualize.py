import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("results/account_value_trade.csv")
value = data["Asset"].to_numpy()
day = np.arange(value.size)

plt.figure(figsize=(8, 6))
# Create a scatter plot
plt.plot(day, value, linewidth=1)
# Label the axes and set a title
plt.xlabel("Trading Day")
plt.ylabel("Asset Value")
plt.title("Viausalization of trading dataset performance")

# Display the plot

plt.show()
