from PCA import SummaryPCA
from Outliers import Outliers
from TSNE import SummaryTSNE
import pandas as pd


def main():
    filepath = '.\\Datasets\\iris.csv'
    col_types = {
        'sepal.length': 'num',
        'sepal.width': 'num',
        'petal.length': 'num',
        'petal.width': 'num',
        'variety': 'char'
    }
    df = pd.read_csv(filepath_or_buffer=filepath, sep=',')
    outliers = Outliers(df, col_types)
    summary = outliers.outlier_summary()
    print(summary)
    pca = SummaryPCA(df, col_types)
    pca.plot_cumulative()
    tsme = SummaryTSNE(df, col_types)
    tsme.plot_target_col('variety')


if __name__ == "__main__":
    main()
