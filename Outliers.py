import pandas as pd 
import numpy as np
from sklearn.ensemble import IsolationForest

class Outliers:
    def __init__(self, df, col_types):
        self.df = df
        self.col_types = col_types
        
    
    # Z-score Detection
    def detect_z_score_col(self, col_name, threshold = 3):
        outlier_count = 0
        cell_count = 0
        mean = np.mean(self.df[col_name])
        std = np.std(self.df[col_name])
        outlier_list = []
        for cell in self.df[col_name]:
            if pd.notnull(cell):
                cell_count += 1
                z_score = (cell - mean) / std            
                if(np.abs(z_score) > threshold):
                    outlier_list.append(1)
                    outlier_count += 1
                else:
                    outlier_list.append(0)
            else:
                outlier_list.append(0)

        outlier_list = np.array(outlier_list)
        return outlier_count, cell_count, outlier_list
    
    def detect_z_score(self, threshold = 3):
        output = {}
        for col_name in self.col_types:
            if(self.col_types[col_name]=='num'):
               outlier_count, cell_count, _ = self.detect_z_score_col(
                   col_name, threshold)
               output[col_name] = (outlier_count, 
                                   round(100*outlier_count/cell_count,2))
        return output
   
    
    # IQR Detection
    def detect_iqr_col(self, col_name, scalar=1.5):
        outlier_count = 0
        cell_count = 0
        quantile1 = np.percentile(self.df[col_name].dropna(),25)
        quantile3 = np.percentile(self.df[col_name].dropna(),75)
        iqr_value = quantile3 - quantile1
        lower_bound = quantile1 - (scalar * iqr_value)
        higher_bound = quantile3 + (scalar * iqr_value)
        outlier_list = []
        
        for cell in self.df[col_name]:
            if pd.notnull(cell):
                cell_count += 1
                if(cell < lower_bound or cell > higher_bound):
                    outlier_list.append(1)
                    outlier_count += 1  
                else:
                    outlier_list.append(0)
            else:
                outlier_list.append(0)
        outlier_list = np.array(outlier_list)
        return outlier_count, cell_count, outlier_list

    def detect_iqr(self, scalar=1.5):
        output = {}
        for col_name in self.col_types:
            if(self.col_types[col_name]=='num'):
               outlier_count, cell_count, _ = self.detect_iqr_col(col_name, 
                                                                  scalar)
               output[col_name] = (outlier_count, 
                                   round(100*outlier_count/cell_count,2))
        return output
    
    
    # Isolation Forest Detection
    # def detect_iforest_col(self, col_name, n_estimators=100, 
    #                        contamination=0.1, max_features=1.0):
        
    #     cell_count = df[col_name].notna().sum()
    #     x = self.df[[col_name]].dropna().to_numpy()
    #     model = IsolationForest(
    #         n_estimators=n_estimators, contamination=contamination, 
    #         max_features=max_features)
    #     model.fit(X=x)
    #     y = model.predict(X=x)
    #     outlier_count = 0
    #     y[y == 1] = 0
    #     y[y == -1] = 1
    #     for val in y:
    #         if val == 1: 
    #             outlier_count += 1
        
    #     return (outlier_count, cell_count, y)
    
    # def detect_iforest(self, n_estimators=100, 
    #                             contamination=0.1, max_features=1.0):
    #     output = {}
    #     for col_name in self.col_types:
    #         if(self.col_types[col_name]=='num'):
    #             outlier_count, cell_count, _ = self.detect_iforest_col(
    #                 col_name, n_estimators, contamination, max_features)
    #             output[col_name] = (outlier_count, 
    #                                 round(100*outlier_count/cell_count,2))
    #     return output

    def real_iforest(self, n_estimators=100, 
                     contamination=0.1, max_features=1.0):
        outlier_count = 0
        data = self.df.copy()
        for key in self.col_types:
            if self.col_types[key] != 'num':
                data.drop(key, axis=1, inplace=True)
        data.dropna(inplace=True)
        model = IsolationForest()
        model.fit(X=data)
        y = model.predict(X=data)
        y[y == 1] = 0
        y[y == -1] = 1
        for val in y:
            if val == 1: 
                outlier_count += 1
        return outlier_count, 100*outlier_count/y.size
            
    
    # Distance Method
    
    def detect_distance_col(self, col_name, contamination=0.02):   
        if(self.col_types[col_name]=='num'):
            
            mean = np.mean(self.df[col_name])
            # Creating array with all column observations with subtracted mean
            arr = np.array(abs(mean - self.df[col_name]))
            num_outliers = round(contamination * len(arr))
            arr = np.nan_to_num(arr, nan=0)
            # Create array with the location of the N biggest elements 
            # in the array where N is the number of outliers
            indices = np.argpartition(arr, -num_outliers)[-num_outliers:]
            outlier_list = np.zeros(len(df[col_name]))
            # Creating output list with 1's where the outliers are located 
            outlier_list[indices] = 1
            return num_outliers, outlier_list
        print('This column is not numeric')
    
    # Main Methods
    def outlier_summary(self, if_n_est=100, if_contamination=0.1,
                        if_max_feat=1.0, iqr_scalar=1.5, z_threshold=3):
        z_tot = 0
        iqr_tot = 0
        iforest_tot = 0
        iforest_perc = 0
        columns = []
        z_score_outliers = []  
        df_out = pd.DataFrame()
        z_score_list = self.detect_z_score(z_threshold)
        iqr_list = self.detect_iqr(iqr_scalar)
        
        # Preparing data for "Columns" column and z-score column
        for key in z_score_list:
            columns.append(key)
            z_score_outliers.append(z_score_list[key])
        
        # Preparing data for the IQR and IForest column 
        iqr_outliers = [iqr_list[key] for key in iqr_list]

        # Filling all the columns with prepared data
        df_out = df_out.assign(
            Columns=columns, Z_Score=z_score_outliers, 
            IQR=iqr_outliers, IForest='-')
        for i in range(len(columns)):
            z_tot += df_out['Z_Score'][i][0]
            iqr_tot += df_out['IQR'][i][0]
        iforest_tot, iforest_perc = self.real_iforest(if_n_est,if_contamination,
                                           if_max_feat)
        
        # Decoration of the output
        df_out['Z_Score'] = df_out['Z_Score'].apply(
            lambda x: str(x[0]) + ' (' + str(x[1]) + '%)')
        df_out['IQR'] = df_out['IQR'].apply(
            lambda x: str(x[0]) + ' (' + str(x[1]) + '%)')
        
        df_out.loc[len(df_out)] = [
            'Total', str(z_tot),str(iqr_tot),str(iforest_tot)]
        
        print('SUMMARY ATTENTION: Isolation Forest calculates outliers \
              excluding NaN values!')
        return df_out
    
    def confusion_matrix(self,if_n_est=100,if_contamination=0.1,
                        if_max_feat=1.0, iqr_scalar=1.5, z_threshold=3):
        z_iqr_res = 0
        z_detect_res = 0
        iqr_detect_res = 0
        z_count = 0
        iqr_count = 0
        distance_count = 0
        df_conf = pd.DataFrame()     
        df_conf = df_conf.assign(Methods=['Z_Score','IQR', 'Distance'])
        
        for col_name in self.col_types:
            if(self.col_types[col_name] == 'num'):
                
                # Getting Data
                z_count_col, _, z_arr = self.detect_z_score_col(
                    col_name, z_threshold)
                iqr_count_col, _, iqr_arr = self.detect_iqr_col(
                    col_name, iqr_scalar)
                distance_count_col, distance_arr = self.detect_distance_col(
                    col_name)
                
                # Preparing and assigning data to Z-Score Column
                z_count += z_count_col
                for i in range(min(len(z_arr),len(iqr_arr))):
                    if z_arr[i] == iqr_arr[i] and z_arr[i] == 1:
                        z_iqr_res += 1
                for i in range(min(len(z_arr),len(distance_arr))):
                    if z_arr[i] == distance_arr[i] and z_arr[i] == 1:
                        z_detect_res += 1
                df_conf = df_conf.assign(Z_Score=[
                    z_count, z_iqr_res, z_detect_res])
                
                # Preparing and assigning data to IQR column
                iqr_count += iqr_count_col
                for i in range(min(len(iqr_arr),len(distance_arr))):
                    if iqr_arr[i] == distance_arr[i] and iqr_arr[i] == 1:
                        iqr_detect_res += 1
                df_conf = df_conf.assign(IQR=[
                    z_iqr_res,iqr_count,iqr_detect_res])
                
                # Preparing and assigning data to Distance column
                distance_count += distance_count_col
                df_conf = df_conf.assign(Distance=[
                    z_detect_res, iqr_detect_res,distance_count])    
        return df_conf
        
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
    outy1 = Outliers(df1, col_types_1)     
    outy2 = Outliers(df2, col_types_2)            
    outy3 = Outliers(df3, col_types_3)            
    outy4 = Outliers(df4, col_types_4)            
    outy5 = Outliers(df5, col_types_5)            
       
    
    summary1 = outy1.outlier_summary()
    summary2 = outy2.outlier_summary()
    summary3 = outy3.outlier_summary()
    summary4 = outy4.outlier_summary()
    summary5 = outy5.outlier_summary()

    print('-------------------------------------------------------------')
    print(summary1)
    print('-------------------------------------------------------------')
    print(summary2)
    print('-------------------------------------------------------------')
    print(summary3)
    print('-------------------------------------------------------------')
    print(summary4)
    print('-------------------------------------------------------------')
    print(summary5)
    print('-------------------------------------------------------------')

    # print()
    #conf_mat = outy.confusion_matrix()
    #print(conf_mat)
    # outy.detect_distance_col('age')
    # c, y = outy.real_iforest()
    # print(y.shape)
    # print(y)
    # print(c)
    # # print(res)
    