from sklearn.preprocessing import MinMaxScaler,RobustScaler


class Scaler_wrapper:
    def __init__(self, scaler) -> None:
        self.scaler = scaler
    
    def fit_transform(self, df_train, df_test_list, feature_map):
        df_list_return = []
        df_train[feature_map] = self.scaler.fit_transform(df_train[feature_map])
        df_list_return.append(df_train)
        for df_test in df_test_list:
            df_test[feature_map] = self.scaler.transform(df_test[feature_map])
            df_list_return.append(df_test)
        return df_list_return

    def fit(self, df_train, feature_map):
        self.scaler.fit(df_train[feature_map])
    
    def transform(self, df, feature_map):
        df[feature_map] = self.scaler.transform(df[feature_map])
        return df

    def inverse(self, df, feature_map):
        df[feature_map] = self.scaler.inverse_transform(df[feature_map])
        return df
    
def scaler_wrapper(train,test,val,feature_map,scaler_type,only_fit = False):
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    elif scaler_type == 'None':
        return train,test,val,None
    else:
        raise Exception('Scaler type not supported')
    scaler_wrapper = Scaler_wrapper(scaler)
    scaler_wrapper.fit(train,feature_map)
    if only_fit:
        return train,test,val,scaler_wrapper
    train = scaler_wrapper.transform(train,feature_map)
    test = scaler_wrapper.transform(test,feature_map)
    val = scaler_wrapper.transform(val,feature_map)
    return train,test,val,scaler_wrapper