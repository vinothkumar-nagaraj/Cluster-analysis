import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Create a sample data frame
data = {
    'Category': ['A', 'B', 'C', 'D'],
    'Values': [10, 20, 15, 25]
}
df = pd.DataFrame(data)

# Streamlit app
st.title('Streamlit Data Frame and Visualization')

# Display the data frame
st.write("Here's our data frame:")
st.dataframe(df)

# Create a bar plot
fig, ax = plt.subplots()
ax.bar(df['Category'], df['Values'], color='skyblue')
ax.set_xlabel('Category')
ax.set_ylabel('Values')
ax.set_title('Bar Plot of Values by Category')

# Display the plot
st.pyplot(fig)
