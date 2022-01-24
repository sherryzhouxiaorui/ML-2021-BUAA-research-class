#用决策树模型对乳腺癌数据集进行分类

from sklearn import tree
#加载sklearn自带数据集
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
cancer = load_breast_cancer()

#查看数据集
#print(cancer.target)

#制表
#print(pd.concat([pd.DataFrame(cancer.data),pd.DataFrame(cancer.target)],axis=1))
#print(cancer.feature_names)

#设置训练集大小
Xtrain,Xtest,Ytrain,Ytest = train_test_split(cancer.data,cancer.target,test_size=0.3)
#print(Xtrain.shape)

#训练模型
clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(Xtrain,Ytrain)
score = clf.score(Xtest,Ytest)

#输出模型得分
print(score)

feature_name=['mean radius','mean texture','mean perimeter','mean area',
 'mean smoothness','mean compactness','mean concavity',
 'mean concave points','mean symmetry','mean fractal dimension',
 'radius error','texture error','perimeter error','area error',
 'smoothness error','compactness error','concavity error',
 'concave points error','symmetry error','fractal dimension error',
 'worst radius','worst texture','worst perimeter','worst area',
 'worst smoothness','worst compactness','worst concavity',
 'worst concave points','worst symmetry','worst fractal dimension']

#引入绘图模块
import graphviz
#设置标签、颜色
dot_data = tree.export_graphviz(clf,feature_names=feature_name,class_names=["恶性","良性"]
                                ,filled=True,rounded=True)
graph = graphviz.Source(dot_data)


#生成相同的樹
clf = tree.DecisionTreeClassifier(criterion="entropy",random_state=30)
clf = clf.fit(Xtrain,Ytrain)
score = clf.score(Xtest,Ytest)
print(score)

#增加隨機性
clf = tree.DecisionTreeClassifier(criterion="entropy",random_state=30,splitter="random")
clf = clf.fit(Xtrain,Ytrain)
score = clf.score(Xtest,Ytest)
print(score)

#樹是否過擬合
score_train = clf.score(Xtrain,Ytrain)
print(score_train)#輕微過擬合，剪枝，更好泛化性

#設置最大深度
clf = tree.DecisionTreeClassifier(criterion="entropy",random_state=30,splitter="random"
                                ,max_depth=3,min_samples_leaf=10,min_samples_split=10)
clf = clf.fit(Xtrain,Ytrain)
score = clf.score(Xtest,Ytest)
print(score)

#不剪枝结果
#dot_data = tree.export_graphviz(clf,feature_names=feature_name,class_names=["恶性","良性"]
                                #,filled=True,rounded=True)
#graph = graphviz.Source(dot_data)

#樹是否過擬合
score_train = clf.score(Xtrain,Ytrain)
print(score_train)#輕微過擬合，剪枝，更好泛化性
#當信息增益小於某一個值，分支就不发生了

#剪枝结果
graph = graphviz.Source(dot_data)
graph.view()


#确定参数——通过画学习曲线
#绘制学习曲线
import matplotlib.pyplot as plt
test = []
for i in range(10):
  clf = tree.DecisionTreeClassifier(max_depth=i+1
                    ,criterion="entropy"
                    ,random_state=30
                    ,splitter="random")
  clf = clf.fit(Xtrain,Ytrain)
  score = clf.score(Xtest,Ytest)
  test.append(score)
plt.plot(range(1,11),test,color="red",label="max_depth")
plt.legend()
plt.show()

#交叉验证
from sklearn.model_selection import cross_val_score
#输出交叉验证得分
print(cross_val_score(clf,cancer.data,cancer.target,cv=10))