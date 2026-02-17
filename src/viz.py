import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

def plot_recency(features):
    sns.histplot(features["recency"], bins=30)
    plt.title("Distribution of Recency")
    plt.xlabel("Days since last purchase")
    plt.show()

def plot_recency_by_churn(df):
    sns.histplot(data=df, x="recency", hue="churn", bins=30, kde=True)
    plt.title("Recency by Churn")
    plt.show()

def plot_corr(df):
    corr = df[["recency", "frequency", "monetary", "churn"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()

def plot_roc(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.show()
