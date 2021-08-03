import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(color_codes=True)

pd.set_option('display.max_columns', 100)

DATA_DIR = os.path.join("./", "ml_case_data")
TRAINING_DATA = os.path.join(DATA_DIR, "ml_case_training_data.csv")
HISTORY_DATA = os.path.join(DATA_DIR, "ml_case_training_hist_data.csv")
CHURN_DATA = os.path.join(DATA_DIR, "ml_case_training_output.csv")

train_data = pd.read_csv(TRAINING_DATA)
churn_data = pd.read_csv(CHURN_DATA)
history_data = pd.read_csv(HISTORY_DATA)

train = pd.merge(train_data, churn_data, on="id")

# pd.DataFrame({"Data type":train.dtypes})


churn = train[["id", "churn"]]
# lets change the 'id' name to 'Companies'
churn.columns = ["Companies", "churn"]

def plot_stacked_bars(dataframe, title_, size_=(18, 10), rot_=0, legend_="upper right"):
    """
    Plot stacked bars with annotations
    """
    ax = dataframe.plot(kind="bar",
                        stacked=True,
                        figsize=size_,
                        rot=rot_,
                        title=title_)
    # Annotate bars
    annotate_stacked_bars(ax, textsize=14)
    # Rename legend
    plt.legend(["Retention", "Churn"], loc=legend_)
    # Labels
    plt.ylabel("Company base (%)")
    plt.show()


def annotate_stacked_bars(ax, pad=0.99, colour="white", textsize=13):
    """
    Add value annotations to the bars
    """
    # Iterate over the plotted rectanges/bars
    for p in ax.patches:
        # Calculate annotation
        value = str(round(p.get_height(), 1))
        # If value is 0 do not annotate
        if value == '0.0':
            continue
        ax.annotate(value, ((p.get_x() + p.get_width() / 2) * pad - 0.05, (p.get_y() + p.get_height() / 2) * pad),
                    color=colour, size=textsize, )


churn_total = churn.groupby(churn["churn"]).count()
churn_percentage = churn_total / churn_total.sum() * 100

plot_stacked_bars(churn_percentage.transpose(), "Churning status", (5, 5), legend_="lower right")
