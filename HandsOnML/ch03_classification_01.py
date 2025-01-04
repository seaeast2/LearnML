from sklearn.datasets import fetch_openml

# 3.1 MNIST
# MNIST 데이터셋을 불러옴
mnist = fetch_openml('mnist_784', version=1)
print(mnist.keys())

# MNIST 데이터셋은 28x28 픽셀 이미지 70,000개로 구성되어 있음
X, y = mnist["data"].to_numpy(), mnist["target"].to_numpy()
print(X.shape)
print(y.shape)

import matplotlib as mpl
import matplotlib.pyplot as plt

some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap=mpl.cm.binary)
plt.axis("off")
plt.show()

print(y[0])

# y는 문자열이므로 정수로 변환
y = y.astype(np.uint8)

# MNIST 데이터셋은 이미 훈련 세트와 테스트 세트로 나누어져 있음
# 훈련 세트는 처음 60,000개 이미지, 테스트 세트는 나머지 10,000개 이미지
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# 3.2 이진 분류기 : 5와 5가 아님으로 분류하는 분류기