import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score



def LoadData():
    '''
        Loading data from data file! 
    '''
    TrainData = pd.read_csv("/home/unbroken/MyFiles/Work/Programming/Kaggle/Titanic/data/train.csv", index_col = 'PassengerId')
    TestData = pd.read_csv("/home/unbroken/MyFiles/Work/Programming/Kaggle/Titanic/data/test.csv", index_col = 'PassengerId')
    Test_label = pd.read_csv("/home/unbroken/MyFiles/Work/Programming/Kaggle/Titanic/data/gender_submission.csv")
    return TrainData, TestData, Test_label

    
def DataProcess(Data):
    '''
        Processing data!
    '''
    Data['Age'].fillna(Data['Age'].median(), inplace=True)  # 将特征年龄数据中的缺失值填充为年龄的中位数
    Data.loc[Data['Sex'] == 'male', 'Sex'] = 1  # 将特征性别数据数值化，male = 1
    Data.loc[Data['Sex'] == 'female', 'Sex'] = 0   # female = 0
    Data.loc[Data['Embarked'] == 'S', 'Embarked'] = 0  # 将特征上船地点数据数值化，S = 0  
    Data.loc[Data['Embarked'] == 'C', 'Embarked'] = 1  # C = 1
    Data.loc[Data['Embarked'] == 'Q', 'Embarked'] = 2  # Q = 2
    Data['Embarked'].fillna(0, inplace = True)
    Data['Fare'].fillna(Data['Fare'].mean(), inplace = True)
    bins = [0, 7, 18, 40, 60, 100]
    Data.Age = pd.cut(Data.Age, bins, labels=[1, 2, 3, 4, 5]) # 将乘客年龄离散化
    return Data
 
 
def FeatureAnalysis(TrainData):
    '''
        Feature analysis!
    '''
    Pclass_1 = TrainData[TrainData.Pclass == 1].Survived
    Pclass_2 = TrainData[TrainData.Pclass == 2].Survived
    Pclass_3 = TrainData[TrainData.Pclass == 3].Survived
    Pclass_label_list = ['Pclass_1', 'Pclass_2', 'Pclass_3']
    Pclass_num_list1 = [sum(Pclass_1), sum(Pclass_2), sum(Pclass_3)]
    Pclass_num_list2 = [len(Pclass_1)-sum(Pclass_1), len(Pclass_2)-sum(Pclass_2), len(Pclass_3)-sum(Pclass_3)]
    Pclass_X = range(len(Pclass_num_list1))
    rect1 = plt.bar(x=Pclass_X, height=Pclass_num_list1, width=0.4, alpha=0.8, color='green')
    rect2 = plt.bar(x=[i+0.4 for i in Pclass_X], height=Pclass_num_list2, width=0.4, alpha=0.8, color='red')
    plt.xticks([index + 0.2 for index in Pclass_X], Pclass_label_list)
    plt.show()
    
    male = TrainData[TrainData.Sex == 1].Survived
    female = TrainData[TrainData.Sex == 0].Survived
    Sex_label_list = ['male', 'female']
    Sex_num_list1 = [sum(male), sum(female)]
    Sex_num_list2 = [len(male) - sum(male), len(female) - sum(female)]
    Sex_X = range(len(Sex_num_list1))
    rect1 = plt.bar(x=Sex_X, height=Sex_num_list1, width=0.4, alpha=0.8, color='green')
    rect2 = plt.bar(x=[i+0.4 for i in Sex_X], height=Sex_num_list2, width=0.4, alpha=0.8, color='red')
    plt.xticks([index + 0.2 for index in Sex_X], Sex_label_list)
    plt.show()
    
    
def LinearRegressionModel(TrainData):
    '''
        Using the linear regression model to predict! 
    '''
    LR_Model = LinearRegression()
    features = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
    kf = KFold(n_splits=3)
    predictions = []
    for train, test in kf.split(TrainData):
        train_predictor = TrainData[features].iloc[train, :]
        train_target = TrainData['Survived'].iloc[train]
        test_predictor = TrainData[features].iloc[test, :]
        LR_Model.fit(train_predictor, train_target)
        test_prediction = LR_Model.predict(test_predictor)
        predictions.extend(test_prediction)
    predictions = np.array(predictions)
    predictions[predictions>0.5] = 1
    predictions[predictions<0.5] = 0
    accuracy_ratio = sum(predictions == TrainData['Survived'])/len(predictions)
    return accuracy_ratio


def LogisticRegressionModel(TrainData, TestData, Test_label):
    '''
        Using the logistic regression model to predict! 
    '''
    features = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
    LogistReg = LogisticRegression(random_state=1)
    scores = cross_val_score(LogistReg, TrainData[features], TrainData['Survived'], cv = 3)
    accuracy_ratio = scores.mean()
    LogistR = LogisticRegression(C = 50, penalty='l1', tol=0.01)
    LogistR.fit(TrainData[features], TrainData['Survived'])
    predict_value = LogistR.predict(TestData[features])
    result = pd.DataFrame({'PassengerId':Test_label.PassengerId,'Survived': pd.Series(predict_value)})
    result.to_csv("/home/unbroken/MyFiles/Work/Programming/Kaggle/Titanic/data/result.csv",index = False)
    return accuracy_ratio
    
    
if __name__ == '__main__':
    TrainData, TestData, Test_label = LoadData()
    TrainData = DataProcess(TrainData)
    TestData = DataProcess(TestData)
#     FeatureAnalysis(TrainData)
    LR_accuracy_ratio = LinearRegressionModel(TrainData)
    LogistReg_accuracy_ratio = LogisticRegressionModel(TrainData, TestData, Test_label)
    
    
    
    
    
    
    
    
    
    