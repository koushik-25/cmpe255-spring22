import pandas as pd
import numpy as np

class CarPrice:

    def __init__(self):
        path = '/Users/koushikpillalamarri/Desktop/SPRING-2022/CMPE-255/LAB-2/data.csv'
        self.df = pd.read_csv(path)
        print(f'${len(self.df)} lines')
    #this function replaces space with underscore
    def trim(self):
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')
        string_columns = list(self.df.dtypes[self.df.dtypes == 'object'].index)
        for col in string_columns:
            self.df[col] = self.df[col].str.lower().str.replace(' ', '_')

    def linear_regression(self, X, y, r=1):
        ones = np.ones(X.shape[0])
        X = np.column_stack([ones, X])
        XTX = X.T.dot(X)
        XTX_inv = np.linalg.inv(XTX)
        w = XTX_inv.dot(X.T).dot(y)
        return w[0], w[1:]

    #Prepare baseline solution
    def prepare_X(self, df,base):
        df_num = df[base]
        df_num = df_num.fillna(0)
        X = df_num.values
        return X
    
    def rmse(self, y, y_pred):
        err = y_pred - y
        mse = (err ** 2).mean()
        return np.sqrt(mse)
    
    def displaying(self, df):
        print(df.iloc[:,5:].head().to_markdown(), "\n")
    
    def validation(self):
        np.random.seed(5)
        n = len(self.df)
        n_val = int(0.2 * n)
        n_test = int(0.2 * n)
        n_train = n - (n_val + n_test)
        idx = np.arange(n)
        np.random.shuffle(idx)
        df_shuffled = self.df.iloc[idx]
        df_train = df_shuffled.iloc[:n_train].copy()
        df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
        df_test = df_shuffled.iloc[n_train+n_val:].copy()
        y_train_orig = df_train.msrp.values
        y_val_orig = df_val.msrp.values
        y_test_orig = df_test.msrp.values
        y_train = np.log1p(df_train.msrp.values)
        y_val = np.log1p(df_val.msrp.values)
        y_test = np.log1p(df_test.msrp.values)
        del df_train['msrp']
        del df_val['msrp']
        del df_test['msrp']
        base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']
        X_train = self.prepare_X(df_train,base)
        X_validation = self.prepare_X(df_val,base)

        w_0, w = self.linear_regression(X_train, y_train)
        y_pred_val = w_0 + X_validation.dot(w)
        print("validation set value", self.rmse(y_val, y_pred_val))

        X_test = self.prepare_X(df_test, base)
        y_pred_test = w_0 + X_test.dot(w)
        print("rmse value of predicted msrp and actual msrp is given by", self.rmse(y_test, y_pred_test))

        y_pred_MSRP_val = np.expm1(y_pred_val) 
        df_val['msrp'] = y_val_orig 
        df_val['msrp_pred'] = y_pred_MSRP_val 

        y_pred_MSRP_test = np.expm1(y_pred_test) 
        df_test['msrp'] = y_test_orig 
        df_test['msrp_pred'] = y_pred_MSRP_test 
        self.displaying(df_val)


if __name__ == "__main__":
    carprice = CarPrice()
    carprice.trim()
    carprice.validation()

