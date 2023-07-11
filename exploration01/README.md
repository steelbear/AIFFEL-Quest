# AIFFEL Campus Online 5th Code Peer Review
- 코더 : 최철웅
- 리뷰어 : 본인의 이름을 작성하세요.


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [X] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  
- [ ] 주석을 보고 작성자의 코드가 이해되었나요?
  > 위 항목에 대한 근거 작성 필수
- [ ] 코드가 에러를 유발할 가능성이 없나요?
  >위 항목에 대한 근거 작성 필수
- [ ] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 위 항목에 대한 근거 작성 필수
- [ ] 코드가 간결한가요?
  > 위 항목에 대한 근거 작성 필수

# 예시
1. 코드의 작동 방식을 주석으로 기록합니다.
2. 코드의 작동 방식에 대한 개선 방법을 주석으로 기록합니다.
3. 참고한 링크 및 ChatGPT 프롬프트 명령어가 있다면 주석으로 남겨주세요.

## 프로젝트 1: 손수 설계하는 선형회귀, 당뇨병 수치를 맞춰보자!
```python
from sklearn.datasets import load_diabetes

diabetes = load_diabetes() # 당뇨병 데이터 불러오기
df_X = diabetes['data']
df_y = diabetes['target']


print(type(df_X)) # 이미 numpy array로 변환되어있음


print(type(df_y)) # 이미 numpy array로 변환되어있음


from sklearn.model_selection import train_test_split


# 데이터셋을 train용과 test용으로 분리
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=42)


print(X_train.shape, y_train.shape) # train 데이터셋 형태 확인
print(X_test.shape, y_test.shape)   # test 데이터셋 형태 확인


import numpy as np

W = np.random.random(10) # 특성 10개에 대한 가중치
b = np.random.random()   # 편향


print(W) # 초기화된 가중치 확인


print(b) # 초기화된 편향 확인


def model(X, W, b):  # X: (batch, features), W: (features,), b: (1, )
    y = X.dot(W) + b
    return y         # y: (batch, )
    
    
# 결과값 크기 확인용
#X = np.random.random((16, 10))
#model(X, W, b).shape


# MSE 계산
def MSE(y_pred, y):
    return ((y_pred - y) ** 2).mean()
  
  
# 파라미터와 데이터셋의 Loss값 계산
def loss(X, W, b, y):
    y_pred = model(X, W, b)
    return MSE(y_pred, y)
    
    
def gradient(X, W, b, y): # X: (batch, features), W: (batch,), b: (1, ), y: (batch, )
    # 그래디언트 연산을 위해 필요한 값
    y_pred = model(X, W, b)
    N = len(y)
    
    # Loss가 MSE인 선형 회귀 모델의 그래디언트 연산
    dW = 1/N * 2 * (X.T.dot(y_pred - y))
    db = 2 * (y_pred - y).mean()
    return dW, db         # dW: (features, ), db: (1, )


# 결과값 크기 확인용
#X = np.random.random((16, 10))
#y = np.random.random(16)

#dW, db = gradient(X, W, b, y)
#print(dW.shape, db.shape)


# 하이퍼 파라미터
LEARNING_RATE = 0.7 # 학습률


losses = []
max_epochs = 10000

# 모델 학습
for i in range(1, max_epochs + 1):
    # 그래디언트 계산
    dW, db = gradient(X_train, W, b, y_train)
    
    # 손실함수 계산
    L = loss(X_train, W, b, y_train)
    losses.append(L) # 각 epoch당 손실함수 저장
    
    # 경사하강법에 따라 파라미터 업데이트
    W -= LEARNING_RATE * dW
    b -= LEARNING_RATE * db
    
    # 중간 결과 출력
    if i % 50 == 0:
        print("Iteration %04i: loss = %.04f" % (i, L))


# epoch에 따른 Loss값 그래프
import matplotlib.pyplot as plt

plt.plot(losses)
plt.title('MSE Losses')
plt.xlabel('epoch')
plt.ylabel('losses')
plt.show()


# test 데이터셋의 실제 값과 모델의 예측값 비교
plt.scatter(X_test[:,0], y_test)
plt.scatter(X_test[:,0], y_test_pred)

plt.legend(['ground truth', 'prediction'])
plt.xlabel('age')
plt.ylabel('disease progression')
plt.show()
```

## 프로젝트 2: 날씨 좋은 월요일 오후 세시, 자전거 타는 사람은 몇명?
```python
import pandas as pd

# train 데이터셋 불러오기
df_train = pd.read_csv('~/aiffel/bike_regression/data/bike-sharing-demand/train.csv')
df_train.head()


# object로 되어있는 'datetime'항목을 datetime으로 변환
s_datetime = pd.to_datetime(df_train['datetime'])

# 년, 월, 일, 시, 분, 초 별로 각 열에 할당
df_train['year'] = s_datetime.map(lambda dt: dt.year)
df_train['month'] = s_datetime.map(lambda dt: dt.month)
df_train['day'] = s_datetime.map(lambda dt: dt.day)
df_train['hour'] = s_datetime.map(lambda dt: dt.hour)
df_train['minute'] = s_datetime.map(lambda dt: dt.minute)
df_train['second'] = s_datetime.map(lambda dt: dt.second)


# 추가한 열 확인
df_train[['year', 'month', 'day', 'hour', 'minute', 'second']]


import matplotlib.pyplot as plt
import seaborn as sns

# 전체 이미지 크기 지정
_ = plt.subplots(figsize=(16, 20))

# 항목별 그래프가 그려질 공간
axes = [plt.subplot(3, 2, i) for i in range(1, 7)]

# 그래프로 그려질 항목
cols = ['year', 'month', 'day', 'hour', 'minute', 'second']

# 항목별 counterplot 그리기
for i in range(6):
    sns.countplot(x=df_train[cols[i]], ax=axes[i])
    axes[i].set_xlabel(cols[i])

plt.show()


# train 데이터셋의 속성 확인
df_train.info()


# 실제로 holiday와 workingday가 배타적 관계가 아님을 확인
(df_train['holiday'] != df_train['workingday']).all()


# 제외한 이유
# datetime: 이미 다른 항목에 구분해 저장
# atemp: temp와 windspeed로 구할 수 있다
# minute, second: 값이 한 종류밖에 없다
# count: 정답 데이터
# casual, registered: 정답 데이터인 count의 세부 항목 (count = casual + registered)
X_cols = ['holiday', 'workingday', 'weather', 'temp', 'humidity', 'windspeed', 'year', 'month', 'day', 'hour']


# 입력 데이터와 정답 데이터를 numpy array로 저장
X = df_train[X_cols].values
y = df_train['count'].values


# 입력 데이터와 정답 데이터 타입과 형태 확인
print(type(X), X.shape)
print(type(y), y.shape)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 데이터셋을 train용과 test용으로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, y_train.shape) # train 데이터셋 형태 확인
print(X_test.shape, y_test.shape)   # test 데이터셋 형태 확인


model = LinearRegression()  # 모델 불러오기
model.fit(X_train, y_train) # 모델 학습


from sklearn.metrics import mean_squared_error

# 학습한 모델로 test 데이터셋 추론
y_test_pred = model.predict(X_test)
print(y_test_pred)


# test 데이터셋에 대한 Loss 값(MSE, RMSE) 계산
mse = mean_squared_error(y_test_pred, y_test)
rmse = mse ** 0.5

print("mse:", mse, "rmse:", rmse)


# 전체 이미지 크기 지정
_ = plt.subplots(figsize=(16, 32))

# temp x count 산포도
ax1 = plt.subplot(2, 1, 1)

ax1.scatter(X_test[:,X_cols.index('temp')], y_test)
ax1.scatter(X_test[:,X_cols.index('temp')], y_test_pred)
ax1.legend(['groud truth', 'prediction'])
ax1.set_xlabel('temp')
ax1.set_ylabel('count')

# humidity x count 산포도
ax2 = plt.subplot(2, 1, 2)

ax2.scatter(X_test[:,X_cols.index('humidity')], y_test)
ax2.scatter(X_test[:,X_cols.index('humidity')], y_test_pred)
ax2.legend(['groud truth', 'prediction'])
ax2.set_xlabel('humidity')
ax2.set_ylabel('count')

plt.show()
```

# 참고 링크 및 코드 개선
```python
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```
