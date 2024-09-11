import pandas as pd
import matplotlib.pyplot as plt

# Create a DataFrame with sample employee data
data = {
    'Employee ID': [1, 2, 3, 4, 5],
    'Name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Williams', 'Chris Brown'],
    'Department': ['Engineering', 'HR', 'Sales', 'Engineering', 'HR'],
    'Salary': [70000, 60000, 75000, 80000, 55000]
}

df = pd.DataFrame(data)

# Plot Salary by Department
plt.figure(figsize=(8, 6))
df.groupby('Department')['Salary'].mean().plot(kind='bar', color='skyblue')

# Add labels and title
plt.title('Average Salary by Department', fontsize=14)
plt.xlabel('Department', fontsize=12)
plt.ylabel('Average Salary ($)', fontsize=12)

# Display the plot
plt.tight_layout()
plt.show()
