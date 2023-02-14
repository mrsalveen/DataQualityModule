import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class SummaryPCA:
    def __init__(self, df, col_types):
        self.df = df
        self.col_types = col_types
        self.id = 0

    def plot_cumulative(self):
        df = self.df.copy(deep=True)
        col_types = self.col_types

        df.dropna(inplace=True)

        for col in col_types:
            if col_types[col] == 'char':
                df.drop(col, axis=1, inplace=True)
        # Standardize the dataframe
        sc = StandardScaler()
        sc.fit(df)
        df = sc.transform(df)

        pca = PCA()
        pca.fit_transform(df)
        exp_var_pca = pca.explained_variance_ratio_
        cum_sum_eigenvalues = np.cumsum(exp_var_pca)
        # Create the visualization plot
        plt.bar(
            range(1, len(exp_var_pca) + 1),
            exp_var_pca,
            alpha=0.6,
            label='Individual explained variance'
        )
        plt.plot(
            range(1, len(cum_sum_eigenvalues) + 1),
            cum_sum_eigenvalues,
            '-ro',
            alpha=0.6,
            label='Cumulative explained variance'
        )
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal component number')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()
        self.id += 1
