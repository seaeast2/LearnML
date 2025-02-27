from sklearn.datasets import fetch_openml
import numpy as np

# 3.1 MNIST
# MNIST 데이터셋을 불러옴
mnist = fetch_openml('mnist_784', version=1)
#print(mnist.keys())

# MNIST 데이터셋은 28x28 픽셀 이미지 70,000개로 구성되어 있음
X, y = mnist["data"].to_numpy(), mnist["target"].to_numpy()
#print(X.shape)
#print(y.shape)

import matplotlib as mpl
import matplotlib.pyplot as plt

some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)

# plt.imshow(some_digit_image, cmap=mpl.cm.binary)
# plt.axis("off")
# plt.show()

# print(y[0])

# y는 문자열이므로 정수로 변환
y = y.astype(np.uint8)

# MNIST 데이터셋은 이미 훈련 세트와 테스트 세트로 나누어져 있음
# 훈련 세트는 처음 60,000개 이미지, 테스트 세트는 나머지 10,000개 이미지
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# 3.2 이진 분류기 : 5와 5가 아님으로 분류하는 분류기
y_train_5 = (y_train == 5)  # 5는 True, 다른 숫자는 False
#print (y_train_5)
y_test_5 = (y_test == 5)
#print (y_test_5)

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

# 숫자 5인지 추측
print(sgd_clf.predict([some_digit]))

# 3.3 성능 측정
# from sklearn.model_selection import StratifiedKFold
# from sklearn.base import clone

# skfolds = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)

# for train_index, test_index in skfolds.split(X_train, y_train_5): # split() 함수를 통하여 훈련 세트를 세 개의 폴드로 나눔
#     clone_clf = clone(sgd_clf)
#     X_train_folds = X_train[train_index]
#     y_train_folds = y_train_5[train_index]
#     X_test_fold = X_train[test_index]
#     y_test_fold = y_train_5[test_index]

#     clone_clf.fit(X_train_folds, y_train_folds) # 모델을 훈련
#     y_pred = clone_clf.predict(X_test_fold)
#     n_correct = sum(y_pred == y_test_fold)
#     print(n_correct / len(y_pred))
    
from sklearn.model_selection import cross_val_score
# print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))

# 5 아님 더미 분류기 만들기
from sklearn.base import BaseEstimator

class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)
    
never_5_clf = Never5Classifier()
print(cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy"))

# 3.3.2 오차 행렬
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_train_5, y_train_pred))

# 3.3.3 정밀도와 재현율
from sklearn.metrics import precision_score, recall_score
# precision_score() : 분류 모델의 성능을 평가
print(precision_score(y_train_5, y_train_pred)) # == 4344 / (4344 + 1307)
# recall_score() : 분류 모델의 성능을 평가
print(recall_score(y_train_5, y_train_pred)) # == 4344 / (4344 + 1077)

# F1 점수 : 정밀도와 재현율의 조화평균(harmonic mean)
from sklearn.metrics import f1_score
print(f1_score(y_train_5, y_train_pred))

