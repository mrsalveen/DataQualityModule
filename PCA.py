import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class ING_PCA:
    def __init__(self, df, col_types, output_folder = None):
        self.df = df
        self.col_types = col_types
        self.output_folder = output_folder
        self.id = 0
        
    def plot_cumulative(self):
        df = self.df.copy(deep = True)
        col_types = self.col_types
        
        df.dropna(inplace = True)
        
        for col in col_types:
            if col_types[col] == 'char':
                df.drop(col, axis=1, inplace = True)
        # Stardardize the dataframe
        sc = StandardScaler()
        sc.fit(df)
        df = sc.transform(df)
        
        pca = PCA()
        pca.fit_transform(df)
        exp_var_pca = pca.explained_variance_ratio_
        cum_sum_eigenvalues = np.cumsum(exp_var_pca)
        # Create the visualization plot
        plt.bar(
            range(1,len(exp_var_pca) + 1), 
            exp_var_pca, 
            alpha=0.6, 
            label='Individual explained variance'
            )
        plt.plot(
            range(1,len(cum_sum_eigenvalues) + 1),
            cum_sum_eigenvalues, 
            '-ro',  
            alpha=0.6, 
            label='Cumulative explained variance'
            )
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal component number')
        plt.legend(loc='best')
        plt.tight_layout()
        self.id += 1
        if self.output_folder is not None:
                plt.savefig(f'{self.output_folder}/PCA_Plot.png', format = 'png', facecolor = 'white')
     
if __name__ == "__main__": 
    
    filepath1 = 'C:\\Users\\WH74JT\\OneDrive - ING\\Downloads\\adult.csv'
    filepath2 = 'C:\\Users\\WH74JT\\OneDrive - ING\\Downloads\\iris.csv'
    filepath3 = 'C:\\Users\\WH74JT\\OneDrive - ING\\Downloads\\bigeastncaabasketball.csv'
    filepath4 = 'C:\\Users\\WH74JT\\OneDrive - ING\\Downloads\\framingham.csv'
    filepath5 = 'C:\\Users\\WH74JT\\OneDrive - ING\\Downloads\\WA_Fn-UseC_-Telco-Customer-Churn.csv'
    col_types_1 = {
        'age': 'num', 
        'workclass': 'char', 
        'fnlwgt': 'num', 
        'education': 'char', 
        'education_num': 'num',
        'marital_status': 'char', 
        'occupation': 'char', 
        'relationship': 'char', 
        'race': 'char', 
        'sex': 'char',
        'capital_gain': 'num', 
        'capital_loss': 'num', 
        'hours_per_week': 'num', 
        'native_country': 'char',
        'income': 'char' ,
    }
    col_types_2 = {
        'sepal.length' : 'num',
        'sepal.width' : 'num',
        'petal.length' : 'num',
        'petal.width' : 'num',
        'variety' : 'char'
        }
    col_types_3 = {  
        'id' : 'num',
        'year' : 'num',
        'rank' : 'num',
        'school' : 'char',
        'games' : 'num',
        'wins' : 'num',
        'losses' : 'num',
        'win_percentage' : 'num',
        'conference_wins' : 'num',
        'conference_losses' : 'num',
        'home_wins' : 'num',
        'home_losses' : 'num',
        'away_wins' : 'num',
        'away_losses' : 'num',
        'offensive_rating' : 'num',
        'defensive_rating' : 'num',
        'net_rating' : 'num',
        'field_goals' : 'num',
        'field_goal_attempts' : 'num',
        'field_goal_percentage' : 'num',
        '3_pointers' : 'num',
        '3_pointer_attempts' : 'num',
        '3_pointer_percentage' : 'num',
        'effective_field_goal_percentage' : 'num',
        'free_throws' : 'num',
        'free_throw_attempts' : 'num',
        'free_throw_percentage' : 'num',
        'offensive_rebounds' : 'num',
        'total_rebounds' : 'num',
        'assists' : 'num',
        'steals' : 'num',
        'blocks' : 'num',
        'turnovers' : 'num',
        'personal_fouls' : 'num',
        'points' : 'num',
        'opponent_points' : 'num',
        'simple_rating' : 'num'
    }
    col_types_4 = {
    'male': 'num', 
    'age': 'num', 
    'education': 'char', 
    'currentSmoker': 'num', 
    'cigsPerDay': 'num', 
    'BPMeds': 'num',
    'prevalentStroke': 'num', 
    'prevalentHyp': 'num', 
    'diabetes': 'num', 
    'totChol': 'num', 
    'sysBP': 'num',
    'diaBP': 'num', 
    'BMI': 'num', 
    'heartRate': 'num', 
    'glucose': 'num', 
    'TenYearCHD': 'num'}
    col_types_5 = {
    'customerID': "char", 
    'gender': 'char', 
    'SeniorCitizen': 'num', 
    'Partner': 'char', 
    'Dependents': 'char',
    'tenure': 'num', 
    'PhoneService': 'char', 
    'MultipleLines': 'char', 
    'InternetService': 'char',
    'OnlineSecurity': 'char', 
    'OnlineBackup': 'char', 
    'DeviceProtection': 'char',
    'TechSupport': 'char', 
    'StreamingTV': 'char', 
    'StreamingMovies': 'char', 
    'Contract': 'char',
    'PaperlessBilling': 'char', 
    'PaymentMethod': 'char', 
    'MonthlyCharges': 'char',
    'TotalCharges': 'char', 
    'Churn': 'char'}
    
    df1 = pd.read_csv(filepath_or_buffer=filepath1, sep=';')
    df2 = pd.read_csv(filepath_or_buffer=filepath2, sep=',')
    df3 = pd.read_csv(filepath_or_buffer=filepath3, sep=',')
    df4 = pd.read_csv(filepath_or_buffer=filepath4, sep=',')
    df5 = pd.read_csv(filepath_or_buffer=filepath5, sep=',')

    output_folder = 'C:\\Users\\WH74JT\\OneDrive - ING\\Desktop\\Plots'
    pca1 = ING_PCA(df1,col_types_1, output_folder = output_folder)
    pca2 = ING_PCA(df2,col_types_2, output_folder = output_folder)
    pca3 = ING_PCA(df3,col_types_3, output_folder = output_folder)
    pca4 = ING_PCA(df4,col_types_4, output_folder = output_folder)
    pca5 = ING_PCA(df5,col_types_5, output_folder = output_folder)
    
    # pca1.plot_cumulative()
    
    # pca2.plot_cumulative()
    
    # pca3.plot_cumulative()
    
    pca4.plot_cumulative()
    
    # pca5.plot_cumulative()
    

