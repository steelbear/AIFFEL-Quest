# AIFFEL Campus Online 5th Code Peer Review
- 코더 : 최철웅
- 리뷰어 : 김범준


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [V] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  > 마지막 코드를 제외하고는 정상적으로 동작하였습니다.
- [V] 주석을 보고 작성자의 코드가 이해되었나요?
  > 각각의 코드에 대해 왜 해당 코드가 작성되었는지를 명확하게 이해하고 계셨고, 이해한 내용을 바탕으로 어떤 맥락에서 이 코드가 작성이 되었는지, 그래서 이 코드가 어떤 작동을 하는지 등 명확하게 설명해 주셨습니다.
- [V] 코드가 에러를 유발할 가능성이 없나요?
  > 마지막의 input shape와 관련한 에러가 발견되었지만 개선 사항으로 인식하고 계셨으며, 이전까지의 코드들의 경우 에러가 발생하지 않았습니다.
- [V] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 마찬가지로 배경과 작동 결과를 이해하고 설명해주셨습니다.
- [V] 코드가 간결한가요?
  > 필요한 코드들이 적절하게 들어간 것으로 보입니다.

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

원문: text
요약문: 

전처리를 하는 과정에서 원문의 단어가 한 개 이상일 경우 결측치로 판단하지 않고 삭제하지 않았습니다. 
결과로, 위의 사례와 같은 경우가 발생하게 되었는데, 단어의 개수가 현저하게 적은 경우도 결측치로 포함해 삭제하는 것이 좋지 않을까하는 이야기를 나누었습니다.
