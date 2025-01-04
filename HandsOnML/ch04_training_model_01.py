# 4.1.1 정규방정식

# 무작위로 선형 데이터 생성
import numpy as np
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 정규 방정식을 사용하여 theta 계산
X_b = np.c_[np.ones((100, 1)), X] # 모든 샘플에 x0 = 1을 추가
# np.linalg.inv() : 넘파이.선형대수모듈.역행렬 계산
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print(theta_best)