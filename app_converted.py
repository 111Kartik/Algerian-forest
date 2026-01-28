import streamlit as st
st.set_page_config(page_title='Converted App', layout='wide')
st.title('Streamlit App (Converted from Colab)')

st.subheader('Section 1')
with st.expander('Show code logic'):
    ## Algerian Forest Fires Dataset
    '''Data Set Information:
    
    The dataset includes 244 instances that regroup a data of two regions of Algeria,namely the Bejaia region located in the northeast of Algeria and the Sidi Bel-abbes region located in the northwest of Algeria.
    
    122 instances for each region.
    
    The period from June 2012 to September 2012.
    The dataset includes 11 attribues and 1 output attribue (class)
    The 244 instances have been classified into fire(138 classes) and not fire (106 classes) classes.
    Attribute Information:
    
    1. Date : (DD/MM/YYYY) Day, month ('june' to 'september'), year (2012)
    Weather data observations
    2. Temp : temperature noon (temperature max) in Celsius degrees: 22 to 42
    3. RH : Relative Humidity in %: 21 to 90
    4. Ws :Wind speed in km/h: 6 to 29
    5. Rain: total day in mm: 0 to 16.8
    FWI Components
    6. Fine Fuel Moisture Code (FFMC) index from the FWI system: 28.6 to 92.5
    7. Duff Moisture Code (DMC) index from the FWI system: 1.1 to 65.9
    8. Drought Code (DC) index from the FWI system: 7 to 220.4
    9. Initial Spread Index (ISI) index from the FWI system: 0 to 18.5
    10. Buildup Index (BUI) index from the FWI system: 1.1 to 68
    11. Fire Weather Index (FWI) Index: 0 to 31.1
    12. Classes: two classes, namely Fire and not Fire '''

st.subheader('Section 2')
with st.expander('Show code logic'):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    %matplotlib inline

st.subheader('Section 3')
with st.expander('Show code logic'):
    dataset=pd.read_csv('Algerian_forest_fires_dataset_UPDATE.csv' ,header=1)
    dataset.head()
    dataset.info()

st.subheader('Section 4')
with st.expander('Show code logic'):
    dataset[dataset.isnull().any(axis=1)]

st.subheader('Section 5')
with st.expander('Show code logic'):
    dataset.loc[:122,"Region"]=0
    dataset.loc[122:,"Region"]=1
    df=dataset
    df.info()

st.subheader('Section 6')
with st.expander('Show code logic'):
    df.head()

st.subheader('Section 7')
with st.expander('Show code logic'):
    df[['Region']]=df[['Region']].astype(int)
    df.head()

st.subheader('Section 8')
with st.expander('Show code logic'):
    df.isnull().sum()

st.subheader('Section 9')
with st.expander('Show code logic'):
    ## Removing the null values
    df=df.dropna().reset_index(drop=True)
    df.head()
    df.isnull().sum()

st.subheader('Section 10')
with st.expander('Show code logic'):
    df.iloc[[122]]

st.subheader('Section 11')
with st.expander('Show code logic'):
    ##remove the 122nd row
    df=df.drop(122).reset_index(drop=True)
    df.iloc[[122]]
    df.columns

st.subheader('Section 12')
with st.expander('Show code logic'):
    ## fix spaces in columns names
    df.columns=df.columns.str.strip()
    df.columns
    df.info()

st.subheader('Section 13')
with st.expander('Show code logic'):
    df[['month','day','year','Temperature','RH','Ws']]=df[['month','day','year','Temperature','RH','Ws']].astype(int)
    df.info()

st.subheader('Section 14')
with st.expander('Show code logic'):
    df.head()

st.subheader('Section 15')
with st.expander('Show code logic'):
    objects=[features for features in df.columns if df[features].dtypes=='O']
    objects

st.subheader('Section 16')
with st.expander('Show code logic'):
    for i in objects:
        if i!='Classes':
            df[i]=df[i].astype(float)

st.subheader('Section 17')
with st.expander('Show code logic'):
    df.info()

st.subheader('Section 18')
with st.expander('Show code logic'):
    df.describe()
    df.head()

st.subheader('Section 19')
with st.expander('Show code logic'):
    df.to_csv('Algerian_forest_fires_cleaned_dataset.csv',index=False)

st.subheader('Section 20')
with st.expander('Show code logic'):
    ## drop day,month and year
    df_copy=df.drop(['day','month','year'],axis=1)
    df_copy.head()

st.subheader('Section 21')
with st.expander('Show code logic'):
    ## categories in classes
    df_copy['Classes'].value_counts()

st.subheader('Section 22')
with st.expander('Show code logic'):
    ## Encoding of the categories in classes
    # Check if 'Classes' column is of object type (string) before encoding
    if df_copy['Classes'].dtype == 'object':
        df_copy['Classes']=np.where(df_copy['Classes'].str.contains('not fire'),0,1)
        print("Classes column encoded successfully.")
    else:
        print("Classes column is already numerical (0 for not fire, 1 for fire).")
    display(df_copy.head())

st.subheader('Section 23')
with st.expander('Show code logic'):
    df_copy.head()

st.subheader('Section 24')
with st.expander('Show code logic'):
    df_copy.tail()

st.subheader('Section 25')
with st.expander('Show code logic'):
    df_copy['Classes'].value_counts()

st.subheader('Section 26')
with st.expander('Show code logic'):
    ## Plot desnity plot for all features
    plt.style.use('seaborn-v0_8')
    df_copy.hist(bins=50,figsize=(20,15))
    plt.show()

st.subheader('Section 27')
with st.expander('Show code logic'):
    ## Percentage for Pie Chart
    percentage=df_copy['Classes'].value_counts(normalize=True)*100
    classlabels=["Fire","Not Fire"]
    plt.figure(figsize=(12,7))
    plt.pie(percentage,labels=classlabels,autopct='%1.1f%%')
    plt.title("Pie Chart of Classes")
    plt.show()

st.subheader('Section 28')
with st.expander('Show code logic'):
    df_copy.corr()

st.subheader('Section 29')
with st.expander('Show code logic'):
    sns.heatmap(df_copy.corr(),annot=True)

st.subheader('Section 30')
with st.expander('Show code logic'):
    sns.boxplot(df['FWI'],color='green')

st.subheader('Section 31')
with st.expander('Show code logic'):
    df.head()

st.subheader('Section 32')
with st.expander('Show code logic'):
    df['Classes']=np.where(df['Classes'].str.contains('not fire'),'not fire','fire')

st.subheader('Section 33')
with st.expander('Show code logic'):
    ## Monthly Fire Analysis
    dftemp=df.loc[df['Region']==1]
    plt.subplots(figsize=(13,6))
    sns.set_style('whitegrid')
    sns.countplot(x='month',hue='Classes',data=df)
    plt.ylabel('Number of Fires',weight='bold')
    plt.xlabel('Months',weight='bold')
    plt.title("Fire Analysis of Sidi- Bel Regions",weight='bold')

st.subheader('Section 34')
with st.expander('Show code logic'):
    ## Monthly Fire Analysis
    dftemp=df.loc[df['Region']==0]
    plt.subplots(figsize=(13,6))
    sns.set_style('whitegrid')
    sns.countplot(x='month',hue='Classes',data=df)
    plt.ylabel('Number of Fires',weight='bold')
    plt.xlabel('Months',weight='bold')
    plt.title("Fire Analysis of Brjaia Regions",weight='bold')

st.subheader('Section 35')
with st.expander('Show code logic'):
    X=df_copy.drop('FWI',axis=1)
    y=df_copy['FWI']

st.subheader('Section 36')
with st.expander('Show code logic'):
    X.head()

st.subheader('Section 37')
with st.expander('Show code logic'):
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

st.subheader('Section 38')
with st.expander('Show code logic'):
    X_train.shape,X_test.shape

st.subheader('Section 39')
with st.expander('Show code logic'):
    X_train.corr()

st.subheader('Section 40')
with st.expander('Show code logic'):
    plt.figure(figsize=(12,6))
    sns.heatmap(X_train.corr(),annot=True)

st.subheader('Section 41')
with st.expander('Show code logic'):
    def correlation(dataset,threshold):
        col_corr=set()
        corr_matrix=dataset.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i,j])>threshold:
                    colname=corr_matrix.columns[i]
                    col_corr.add(colname)
        return col_corr

st.subheader('Section 42')
with st.expander('Show code logic'):
    corr_features=correlation(X_train,0.85)
    print(corr_features)

st.subheader('Section 43')
with st.expander('Show code logic'):
    # drop the features when correleation is greater than threshold 0.85
    X_train.drop(corr_features,axis=1,inplace=True, errors='ignore')
    X_test.drop(corr_features,axis=1,inplace=True, errors='ignore')
    X_train.shape,X_test.shape

st.subheader('Section 44')
with st.expander('Show code logic'):
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    X_train_scaled

st.subheader('Section 45')
with st.expander('Show code logic'):
    plt.subplots(figsize=(15,5))
    plt.subplot(1,2,1)
    sns.boxplot(data=X_train['Temperature'])
    plt.title('X_train before scaling')
    plt.subplot(1,2,2)
    sns.boxplot(data=X_train_scaled)
    plt.title('X_train after scaling')

st.subheader('Section 46')
with st.expander('Show code logic'):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_absolute_error
    linreg=LinearRegression()
    linreg.fit(X_train_scaled,y_train)
    y_pred=linreg.predict(X_test_scaled)
    print("R2 Score:",r2_score(y_test,y_pred))
    print("Mean Absolute Error:",mean_absolute_error(y_test,y_pred))
    plt.scatter(y_test,y_pred)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')

st.subheader('Section 47')
with st.expander('Show code logic'):
    from sklearn.linear_model import Lasso
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import r2_score
    lasso=Lasso()
    lasso.fit(X_train_scaled,y_train)
    y_pred=lasso.predict(X_test_scaled)
    print("R2 Score:",r2_score(y_test,y_pred))
    print("Mean Absolute Error:",mean_absolute_error(y_test,y_pred))
    plt.scatter(y_test,y_pred)
    plt.xlabel('Actual Values')

st.subheader('Section 48')
with st.expander('Show code logic'):
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import r2_score
    ridge=Ridge()
    ridge.fit(X_train_scaled,y_train)
    y_pred=ridge.predict(X_test_scaled)
    print("mean absolute error:",mean_absolute_error(y_test,y_pred))
    print("R2 Score:",r2_score(y_test,y_pred))
    plt.scatter(y_test,y_pred)

st.subheader('Section 49')
with st.expander('Show code logic'):
    from sklearn.linear_model import RidgeCV
    ridgecv=RidgeCV(cv=5)
    ridgecv.fit(X_train_scaled,y_train)
    y_pred=ridgecv.predict(X_test_scaled)
    print("mean absolute error:",mean_absolute_error(y_test,y_pred))
    print("r2 score:",r2_score(y_test,y_pred))
    plt.scatter(y_test,y_pred)

st.subheader('Section 50')
with st.expander('Show code logic'):
    from sklearn.linear_model import ElasticNet
    elastic=ElasticNet()
    elastic.fit(X_train_scaled,y_train)
    y_pred=elastic.predict(X_test_scaled)
    print("mean absolute error:",mean_absolute_error(y_test,y_pred))
    print("r2 score:",r2_score(y_test,y_pred))
    plt.scatter(y_test,y_pred)

st.subheader('Section 51')
with st.expander('Show code logic'):

st.subheader('Section 52')
with st.expander('Show code logic'):

st.subheader('Section 53')
with st.expander('Show code logic'):

st.subheader('Section 54')
with st.expander('Show code logic'):

st.subheader('Section 55')
with st.expander('Show code logic'):

st.subheader('Section 56')
with st.expander('Show code logic'):

st.subheader('Section 57')
with st.expander('Show code logic'):

st.subheader('Section 58')
with st.expander('Show code logic'):

st.subheader('Section 59')
with st.expander('Show code logic'):

st.subheader('Section 60')
with st.expander('Show code logic'):

st.subheader('Section 61')
with st.expander('Show code logic'):

st.subheader('Section 62')
with st.expander('Show code logic'):
