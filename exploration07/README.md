# AIFFEL Campus Online 5th Code Peer Review
- 코더 : 최철웅
- 리뷰어 : 양주영


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [X] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  > 노드 이외의 argumentation 함수를 추가해 적용하여 결과를 반환했습니다.
   ![7 철웅님 문제 해결](https://github.com/steelbear/AIFFEL-Quest/assets/134067511/3e638773-24fe-4ee7-9a72-4a9ca1a1b307)

- [X] 주석을 보고 작성자의 코드가 이해되었나요?
  >긴 코드에 주석을 달아주셔서 흐름을 잘 따라갈 수 있었습니다. 
  ![7 철웅님 주석](https://github.com/steelbear/AIFFEL-Quest/assets/134067511/edd75539-821e-4909-a2d6-093c8a993d29)
- [X] 코드가 에러를 유발할 가능성이 없나요?
  > 불필요한 반복이 없고, 객체 구분을 잘 해놓으셔서 에러가 유발될 가능성은 매우 적어보입니다.
- [X] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 코드에 대해 노드에 없는 설명을 달아놓으신걸로 보아 잘 이해하고 계신걸로 보입니다.
  > ![7 철웅님 간결](https://github.com/steelbear/AIFFEL-Quest/assets/134067511/c60bafea-9510-449f-ab4a-c1f6558a5808)
- [X] 코드가 간결한가요?
  > 함수를 잘 정의하여 활용하였습니다.
  ![7 철웅님 에러](https://github.com/steelbear/AIFFEL-Quest/assets/134067511/f5f564e3-3741-4a0b-a6f3-af10dea8661b)


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
