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

activity = train[["id", "activity_new", "churn"]]

activity = activity.groupby([activity["activity_new"], activity["churn"]])["id"].count().unstack(level=1).sort_values(
    by=[0], ascending=False)

activity.plot(kind="bar",
              figsize=(18, 10),
              width=2,
              stacked=True,
              title="SME Activity")
# Labels
plt.ylabel("Number of companies")
plt.xlabel("Activity")
# Rename legend
plt.legend(["Retention", "Churn"], loc="upper right")
# Remove the label for the xticks as the categories are encoded and we can't draw any meaning from them yet
plt.xticks([])
plt.show()

activity_total = activity.fillna(0)[0] + activity.fillna(0)[1]
activity_percentage = activity.fillna(0)[1] / (activity_total) * 100
pd.DataFrame({"Percentage churn": activity_percentage,
              "Total companies": activity_total}).sort_values(by="Percentage churn",
                                                              ascending=False).head(10)

channel = train[["id", "channel_sales", "churn"]]

channel = channel.groupby([channel["channel_sales"],
                           channel["churn"]])["id"].count().unstack(level=1).fillna(0)

channel_churn = (channel.div(channel.sum(axis=1), axis=0) * 100).sort_values(by=[1], ascending=False)

plot_stacked_bars(channel_churn, "Sales Channel", rot_=30)

channel_total = channel.fillna(0)[0] + channel.fillna(0)[1]
channel_percentage = channel.fillna(0)[1] / (channel_total) * 100
pd.DataFrame({"Churn percentage": channel_percentage,
              "Total companies": channel_total}).sort_values(by="Churn percentage",
                                                             ascending=False).head(10)

consumption = train[["id", "cons_12m", "cons_gas_12m", "cons_last_month", "imp_cons", "has_gas", "churn"]]

def plot_distribution(dataframe, column, ax, bins_=50):
    """
    Plot variable distirbution in a stacked histogram of churned or retained company
    """
    # Create a temporal dataframe with the data to be plot
    temp = pd.DataFrame({"Retention": dataframe[dataframe["churn"] == 0][column],
                         "Churn": dataframe[dataframe["churn"] == 1][column]})
    # Plot the histogram
    temp[["Retention", "Churn"]].plot(kind='hist', bins=bins_, ax=ax, stacked=True)
    # X-axis label
    ax.set_xlabel(column)
    # Change the x-axis to plain style
    ax.ticklabel_format(style='plain', axis='x')

