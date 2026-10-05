{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "80acf645-5d9f-4861-a57f-05e3a292ea99",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import seaborn as sn\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.metrics import r2_score\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Load dataset\n",
    "df = pd.read_csv(\"homeprices.csv\")\n",
    "\n",
    "# Scatter plot of area vs price\n",
    "sn.scatterplot(data=df, x='area', y='price')\n",
    "plt.title(\"House Prices vs Area\")\n",
    "plt.show()\n",
    "\n",
    "# Train the model\n",
    "reg = LinearRegression()\n",
    "reg.fit(df[['area']], df['price'])\n",
    "\n",
    "# Predict for 3300 sq.ft.\n",
    "prediction_3300 = reg.predict(pd.DataFrame({'area': [3300]}))\n",
    "print(\"Price for 3300 sq.ft:\", prediction_3300[0])\n",
    "\n",
    "# Slope and intercept\n",
    "print(\"Slope:\", reg.coef_[0])\n",
    "print(\"Intercept:\", reg.intercept_)\n",
    "\n",
    "# Manual calculation for 3300 sq.ft.\n",
    "y_manual = reg.coef_[0] * 3300 + reg.intercept_\n",
    "print(\"Manual Prediction:\", y_manual)\n",
    "\n",
    "# Predict for 5000 sq.ft.\n",
    "prediction_5000 = reg.predict(pd.DataFrame({'area': [5000]}))\n",
    "print(\"Price for 5000 sq.ft:\", prediction_5000[0])\n",
    "\n",
    "# R² Score\n",
    "y_original = df['price']\n",
    "y_predict = reg.predict(df[['area']])\n",
    "R_square = r2_score(y_original, y_predict)\n",
    "print(\"R² Score:\", R_square)\n",
    "\n",
    "# Regression line plot\n",
    "sn.lmplot(data=df, x='area', y='price', ci=None, line_kws={'color':'red'})\n",
    "plt.title(\"Linear Regression Fit\")\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "509800b5-e09e-4158-9bb9-5fb8e4f839ba",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
