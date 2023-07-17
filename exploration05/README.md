# AIFFEL Campus Online 5th Code Peer Review
- 코더 : 최철웅
- 리뷰어 : 홍수정

# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [X] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  > 인물모드의 사람, 동물 적용 및 배경 합성을 수행하였음
- [X] 주석을 보고 작성자의 코드가 이해되었나요?
  > 매 스텝에서 코드 작성의 이유를 적어놓았기 때문에 이해가 어렵지 않음
- [X] 코드가 에러를 유발할 가능성이 없나요?
  > 가능성 있는 부분에 대하여 if문 처리를 통하여 에러가 발생하지 않도록 함
  
```
    # 라벨에 해당하는 컬러값 찾기
    # 만약 해당하는 라벨이 존재하지 않거나 이미지에서 검출되지 않으면 (255,255,255)로 지정
    if label in LABEL_NAMES and label in found_labels: 
        seg_color = tuple(reversed(colormap[LABEL_NAMES.index(label)]))
    else:
        seg_color = (255, 255, 255)
```
- [X] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > label을 찾는 과정에서 reversed를 통해 채널을 맞추는 것을 통해 image channel에 대한 이해도를 알 수 있었음
- [X] 코드가 간결한가요?
  > 함수화하여 불필요한 코드의 중복을 방지함
```
    def get_image_blur(img_path, label, show_mask=False):
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
# label을 인자로 받아 segmentation 영역을 분리하는 과정에서 오타나 없는 label가 입력되어 else로 처리되면
# 굳이 blurring 하지 않아도 되는데 진행을 하게 되어 리소스가 낭비되는 부분이 있지 않을까 해서
# try문으로 처리 해도 되지 않을까 싶습니다.
    try:
        seg_color = tuple(reversed(colormap[LABEL_NAMES.index(label)]))
    except:
        print(f"{label} is NOT included in segmentation contents!")
```
