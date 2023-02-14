import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt


class SummaryTSNE:
    def __init__(self, df, col_types):
        self.df = df
        self.col_types = col_types
        self.id = 0

    def plot_target_col(self, target_col, title='Plot'):
        self.id += 1
        df = self.df.copy(deep=True)
        col_types = self.col_types
        # Remove all NaN
        df.dropna(inplace=True)
        # Set target and remove it from the dataset
        del col_types[target_col]
        y = df[target_col]
        df.drop(target_col, axis=1, inplace=True)
        # Remove all char columns
        for col in col_types:
            if col_types[col] == 'char':
                df.drop(col, axis=1, inplace=True)
        tsne = TSNE(n_components=2, random_state=0)
        z = tsne.fit_transform(df)
        temp_df = pd.DataFrame()
        temp_df["y"] = y
        temp_df["comp-1"] = z[:, 0]
        temp_df["comp-2"] = z[:, 1]
        # Here we calculate how many distinct values there exist in y
        n_distinct = 0
        s = dict()
        arr = temp_df['y'].tolist()
        for i in range(temp_df['y'].size):
            # If not present, then put it in
            # dictionary and print it
            if arr[i] not in s.keys():
                s[arr[i]] = arr[i]
                n_distinct += 1
        sns.scatterplot(x="comp-1", y="comp-2", data=temp_df,
                        hue=temp_df.y.tolist(),
                        palette=sns.color_palette("hls", n_distinct)).set(
            title=title)
        # Put the legend outside the plot area                    
        plt.legend(bbox_to_anchor=(1.02, 1), loc='best', borderaxespad=0)
        plt.show()

    def plot(self, title='Plot'):
        df = self.df.copy(deep=True)
        col_types = self.col_types
        # Remove all NaN
        df.dropna(inplace=True)
        # Remove all char columns
        for col in col_types:
            if col_types[col] == 'char':
                df.drop(col, axis=1, inplace=True)
        tsne = TSNE(n_components=2, random_state=0)
        z = tsne.fit_transform(df)
        temp_df = pd.DataFrame()
        temp_df["comp-1"] = z[:, 0]
        temp_df["comp-2"] = z[:, 1]

        sns.scatterplot(x="comp-1", y="comp-2", data=temp_df).set(title=title)
