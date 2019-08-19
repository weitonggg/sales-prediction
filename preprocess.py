import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(filename):
    df = pd.read_csv(filename)
    return df

def data_preprocess(df):
    
    ## imputation for Item Weights ##
    item_weights = pd.DataFrame(df.pivot_table(values='Item_Weight', index='Item_Identifier'))
    missing = df['Item_Weight'].isnull() 
    df.loc[missing,'Item_Weight'] = df.loc[missing,'Item_Identifier'].apply(lambda x: item_weights[item_weights.index == x].Item_Weight[0])
    missing = df.index[np.isnan(df.Item_Weight)].tolist()
    df = df.drop(missing, axis = 0)
    
    ## imputation for Outlet Size ##
    #outlet_size = pd.DataFrame(pd.crosstab(df.Outlet_Size, df.Outlet_Type))
    missing = df.Outlet_Size.isnull()
    
    for i in df[missing].index:
        if (df.loc[i,'Outlet_Type'] == 'Grocery Store'):
            df.loc[i, 'Outlet_Size'] = 'Small'
        elif df.loc[i, 'Outlet_Type'] == 'Supermarket Type1':
            df.loc[i, 'Outlet_Size'] = 'Small'
        else: df.loc[i, 'Outlet_Size'] = 'Medium'
    
    ## create new feature: Outlet_Age ##
    df['Outlet_Age'] = 2013 - df.Outlet_Establishment_Year
    
    ## revalue items with 0 visibility in outlet ##
    df.Item_Visibility[df.Item_Visibility == 0] = df.Item_Visibility.mean()
    
    ## rename Item Fat Content ##
    df.Item_Fat_Content = df.Item_Fat_Content.replace({'low fat': 'Low Fat',
                                'LF': 'Low Fat',
                                'reg': 'Regular'})
    
    ## regroup Item Type to contain 3 categories ##
    df['Item_Type_new'] = df.Item_Identifier.apply(lambda x: x[0:2])
    df.Item_Type_new = df.Item_Type_new.replace({'FD': 'Food',
                                                     'DR': 'Drinks',
                                                     'NC': 'Non-consumables'})
    
    ## one-hot encoding ##
    fat_content_dummy = pd.get_dummies(df.Item_Fat_Content).rename(columns = lambda x: 'fat_content_' + str(x))
    outlet_iden_dummy = pd.get_dummies(df.Outlet_Identifier).rename(columns = lambda x: 'outlet_iden_' + str(x))
    outlet_size_dummy = pd.get_dummies(df.Outlet_Size).rename(columns = lambda x: 'outlet_size_' + str(x))
    outlet_loctype_dummy= pd.get_dummies(df.Outlet_Location_Type).rename(columns = lambda x: 'outlet_loctype_' + str(x))
    outlet_type_dummy = pd.get_dummies(df.Outlet_Type).rename(columns = lambda x: 'outlet_type_' + str(x))
    item_type_dummy = pd.get_dummies(df.Item_Type_new).rename(columns = lambda x: 'item_type_' + str(x))
    
    
    df = pd.concat([df, item_type_dummy, fat_content_dummy, outlet_iden_dummy, outlet_size_dummy,
                outlet_loctype_dummy, outlet_type_dummy], axis = 1)
    df = df.drop(['Item_Identifier', 'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 
              'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type',
              'Outlet_Type', 'Item_Type_new'], axis = 1)
    
    return df

def split_data(data, target, test_size):
    x = data.drop([target], axis= 1)
    y = data[target]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state=0)
    return x_train, x_test, y_train, y_test


file = 'retail_sales.txt'
df = load_data(file)
df = data_preprocess(df)
x_train, x_test, y_train, y_test = split_data(df, 'Item_Outlet_Sales', 0.3)

