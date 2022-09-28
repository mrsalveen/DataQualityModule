import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
class ING_TSNE:
    def __init__(self, df, col_types, output_folder = None):
        self.df = df
        self.col_types = col_types
        self.output_folder = output_folder
        self.id = 0
        
    def plot_target_col(self, target_col, title='Plot'):
        self.id += 1
        df = self.df.copy(deep = True)
        col_types = self.col_types
        
        # Remove all NaN
        df.dropna(inplace = True)
        
        # Set target and remove it from the dataset
        del col_types[target_col]        
        y = df[target_col]
        df.drop(target_col, axis=1, inplace = True)
        
        # Remove all char columns
        for col in col_types:
            if col_types[col] == 'char':
                df.drop(col, axis=1, inplace = True)
        tsne = TSNE(n_components=2, random_state=0)
        z = tsne.fit_transform(df)
        
        temp_df = pd.DataFrame()
        temp_df["y"] = y
        temp_df["comp-1"] = z[:,0]
        temp_df["comp-2"] = z[:,1]
    
        # Here we calculate how many distinct values there exist in y
        n_distinct = 0
        s = dict();
        arr = temp_df['y'].tolist()
        for i in range(temp_df['y'].size):
            # If not present, then put it in
            # dictionary and print it
            if (arr[i] not in s.keys()):
                s[arr[i]] = arr[i]
                n_distinct += 1
        
        sns.scatterplot(x="comp-1", y="comp-2", data=temp_df, 
                        hue=temp_df.y.tolist(), 
                        palette=sns.color_palette("hls", n_distinct)).set(
                            title=title)
                            
        # Put the legend outside the plot area                    
        plt.legend(bbox_to_anchor=(1.02, 1), loc='best', borderaxespad=0)
        if self.output_folder is not None:
                plt.savefig(f'{self.output_folder}/TSNE_Plot.png', format = 'png', facecolor = 'white')
        
        
    def plot(self, title='Plot', output_folder = None):
        df = self.df.copy(deep = True)
        col_types = self.col_types
        
        # Remove all NaN
        df.dropna(inplace = True)
        
        # Remove all char columns
        for col in col_types:
            if col_types[col] == 'char':
                df.drop(col, axis=1, inplace = True)
        tsne = TSNE(n_components=2, random_state=0)
        z = tsne.fit_transform(df)
        
        temp_df = pd.DataFrame()
        temp_df["comp-1"] = z[:,0]
        temp_df["comp-2"] = z[:,1]
    
        sns.scatterplot(x="comp-1", y="comp-2", data=temp_df).set(title=title)
        if self.output_folder is not None:
                plt.savefig(f'{self.output_folder}/TSME_Plot.png', format = 'png', facecolor = 'white')
        
        
    
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

    tsme1 = ING_TSNE(df1, col_types_1, output_folder= 'C:\\Users\\WH74JT\\OneDrive - ING\\Desktop\\Plots')
    tsme2 = ING_TSNE(df2, col_types_2, output_folder= 'C:\\Users\\WH74JT\\OneDrive - ING\\Desktop\\Plots')
    tsme3 = ING_TSNE(df3, col_types_3, output_folder= 'C:\\Users\\WH74JT\\OneDrive - ING\\Desktop\\Plots')
    tsme4 = ING_TSNE(df4, col_types_4, output_folder= 'C:\\Users\\WH74JT\\OneDrive - ING\\Desktop\\Plots')
    tsme5 = ING_TSNE(df5, col_types_5, output_folder= 'C:\\Users\\WH74JT\\OneDrive - ING\\Desktop\\Plots')
    
    # tsme1.plot_target_col(target_col='age')
    
    # tsme2.plot_target_col('variety')
    
    tsme3.plot_target_col('win_percentage')
    
    # tsme4.plot_target_col('diabetes')
    
    # tsme5.plot_target_col('StreamingMovies')
