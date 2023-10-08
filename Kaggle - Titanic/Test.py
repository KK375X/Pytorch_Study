import pandas

# 导入线性回归的方法库，用来做二分类
from sklearn.linear_model import LinearRegression
# 利用 KFold 做交叉验证
from sklearn.model_selection import KFold

from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
import numpy as np


"""
读取数据
"""
titanic = pandas.read_csv('data/train.csv')


"""
对数据进行预处理，缺失值填充，数值转换等
"""
# 为缺失数据进行补充，补充的数值为该特征列的平均值
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())

# 对性别属性的值进行转换，male=0，female=1
titanic.loc[titanic['Sex'] == 'male', 'Sex'] = 0
titanic.loc[titanic['Sex'] == 'female', 'Sex'] = 1

# 查看这一列中不同的数据值是什么
titanic['Sex'].unique()


# 对上船地点做上述操作，补充缺失值和数值转换
# 对缺失的上船地点用 S 填充，谁多用谁填充原则
titanic['Embarked'] = titanic['Embarked'].fillna('S')

# 令 S=0, C=1, Q=2
titanic.loc[titanic['Embarked'] == 'S', 'Embarked'] = 0
titanic.loc[titanic['Embarked'] == 'C', 'Embarked'] = 1
titanic.loc[titanic['Embarked'] == 'Q', 'Embarked'] = 2


"""
回归模型
"""
# 选取数据集中的训练特征
predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# 导入线性回归模块
alg = LinearRegression()

# 交叉验证
# titanic.shape[0] 获取样本的总数
# n_splits=3 三次交叉验证
kf = KFold(n_splits=3, shuffle=True, random_state=1)

# 训练模型
# 存放预测的结果集
prediction_list = []

# 上述 n_splits=3，表示这里做三次循环，进行三次交叉验证，使用 spilt(titanic) 将数据集分成三份进行迭代
for train, test in kf.split(titanic):
    # 拿出需要训练的数据
    train_predictors = (titanic[predictors].iloc[train, :])
    # 拿出 label 做结果的对比
    train_target = titanic['Survived'].iloc[train]
    # 线性回归应用到上述数据上
    alg.fit(train_predictors, train_target)

    # 使用上述训练过的模型对 test 进行预测
    test_predictors = alg.predict(titanic[predictors].iloc[test, :])

    # 存放预测结果
    prediction_list.append(test_predictors)


"""
对结果进行拟合，使其为0或1
"""
prediction_list = np.concatenate(prediction_list, axis=0)

prediction_list[prediction_list > .5] = 1
prediction_list[prediction_list <= .5] = 0

# 求出模型训练样本里有多少和原数据的结果有多少相同
accuracy = sum(prediction_list[prediction_list == titanic['Survived']]) / len(prediction_list)
print(accuracy)











