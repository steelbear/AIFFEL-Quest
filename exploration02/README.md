# AIFFEL Campus Online 5th Code Peer Review
- 코더 : 최철웅
- 리뷰어 : 홍수정


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [X] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  
- [X] 주석을 보고 작성자의 코드가 이해되었나요?
  > 왜 진행하는 지 각 코드 시작마다 설명을 하기 때문에 이해할 수 있었음
- [X] 코드가 에러를 유발할 가능성이 없나요?
  > 진행에서 필수적인 부분은 명확하게 정의하여 넘어갔기 때문에 에러가 발생할 가능성은 적다고 판단함
- [X] scaler를 사용하여 입력 데이터를 정규화하여 성능의 향상을 도모함
```python
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
- [X] 코드가 간결한가요?
- kwargs를 사용하여 기존의 작성된 코드를 재사용함
```python
models = [
    {
        'name': 'GradientBoosting',
        'model': GradientBoostingRegressor(random_state=random_state, **gb_params)
    },]
```

# 예시
1. 코드의 작동 방식을 주석으로 기록합니다.
2. 코드의 작동 방식에 대한 개선 방법을 주석으로 기록합니다.
3. 참고한 링크 및 ChatGPT 프롬프트 명령어가 있다면 주석으로 남겨주세요.
```python
# 사칙 연산 계산기
class calculator:
    # 예) init의 역할과 각 매서드의 의미를 서술
    def __init__(self, first, second):
        self.first = first
        self.second = second
    
    # 예) 덧셈과 연산 작동 방식에 대한 서술
    def add(self):
        result = self.first + self.second
        return result

a = float(input('첫번째 값을 입력하세요.')) 
b = float(input('두번째 값을 입력하세요.')) 
c = calculator(a, b)
print('덧셈', c.add()) 
```

# 참고 링크 및 코드 개선
```python
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```
