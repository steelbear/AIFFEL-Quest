# AIFFEL Campus Online 5th Code Peer Review
- 코더 : 최철웅
- 리뷰어 : 홍수정


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [ ] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  > 여러 사람이 들어간 사진이지만 스티커의 투명도 조절 이나 다양한 얼굴 각도 혹은 다양한 상황에 대한 실험이 진행되지 않음 
- [X] 주석을 보고 작성자의 코드가 이해되었나요?
  > 각 코드가 어떻게 진행되는 지에 따른 주석이 추가되었음
  > 필요에 따라 각 라인별로 코드에 대한 설명이 들어가 이해가 쉬웠음
```        
        # 스티커 이미지를 사용자 얼굴에 맞춰 크기 조정
        img_sticker = cv2.resize(img_sticker, (w, h))

        # 스티커 이미지 회전
        img_sticker = 255 - img_sticker                                   # 스티커 이미지 색반전
        M_rotate = cv2.getRotationMatrix2D((w // 2, h // 2), theta, 1.0 ) # 회전 행렬 구하기
        img_sticker = cv2.warpAffine(img_sticker, M_rotate, (w, h))       # 스티커에 회전 행렬 적용
        img_sticker = 255 - img_sticker                                   # 스티커 이미지 다시 색반전
```
- [X] 코드가 에러를 유발할 가능성이 없나요?
  > os.getenv('HOME')를 주로 workspace로 사용하지는 않을거 같아 주의가 필요하지만
  > 타 repo의 데이터를 사용하면서 directory에 대한 변경은 필수적으로 체크하고 지나가는 부분이므로
  > 이 부분은 해당 되지 않는다고 판단함
```
# review를 진행하기 위해 colab환경에서 './' 로컬환경으로 변경하여 사용

# 이미지 디렉토리
image_dir = './' #os.getenv('HOME') + '/aiffel/camera_sticker/images/'
# 모델 디렉토리
model_dir = './' #os.getenv('HOME') + '/aiffel/camera_sticker/models/'
```
- [X] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
```
    for dlib_rect, landmark in zip(dlib_rects, list_landmarks):
        # 얼굴이 회전한 각도를 계산
        nose_horizontal = (landmark[35][0] - landmark[31][0],           # 코의 수평선 벡터 
                           landmark[35][1] - landmark[31][1])
        cos_ = np.dot((1,0), nose_horizontal) / (norm(nose_horizontal)) # x축과 수평선 사이의 코사인 계산
        theta = np.arccos(cos_) / np.pi * 180                           # 두 선 사이의 각도 계산
        
        # 코의 수평선 벡터 방향에 맞춰 회전 방향 변경
        if nose_horizontal[1] > 0:
            theta = -theta

        # 스티커 이미지 회전
        img_sticker = 255 - img_sticker                                   # 스티커 이미지 색반전
        M_rotate = cv2.getRotationMatrix2D((w // 2, h // 2), theta, 1.0 ) # 회전 행렬 구하기
        img_sticker = cv2.warpAffine(img_sticker, M_rotate, (w, h))       # 스티커에 회전 행렬 적용
        img_sticker = 255 - img_sticker                                   # 스티커 이미지 다시 색반전
```
  > 위와 같이 코와 수평선을 기준으로 회전한 각도를 계산하여 스티커에 적용함으로써 자연스럽게 스티커 필터를 적용시킴
- [X] 코드가 간결한가요?
  > 좌표를 변수에 저장하여 사용함으로써 img 인덱싱 시에 코드가 깔끔해지고 이해가 쉬워짐
```
        # 스티커가 붙을 영역의 좌표 범위 구하기
        x_min = refined_x
        x_max = refined_x + img_sticker.shape[1]
        y_min = refined_y
        y_max = refined_y + img_sticker.shape[0]
            
        # 스티커가 붙을 영역을 가져오기
        sticker_area = img[y_min:y_max, x_min:x_max]
   
        if weighted : # 불투명한 스티커를 원할 떄
          img[y_min:y_max, x_min:x_max] = \
              cv2.addWeighted(sticker_area, 0.5, np.where(img_sticker==255,sticker_area,img_sticker).astype(np.uint8), 0.5, 0)
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
# "weighted"를 옵션으로 넣어 투명함 여부를 선택할 수 있도록 하면 어떨까 제안드립니다.
# 수치를 조절할 수 있는 인자를 넣을 수 도 있지만 0.5를 기본으로 투명하게하는 기능을 추가하였습니다.

def attach_sticker(img, img_sticker_src, rects, list_landmarks, weighted = False):
 (중략)
        if weighted : # 불투명한 스티커를 원할 떄
          img[y_min:y_max, x_min:x_max] = \
              cv2.addWeighted(sticker_area, 0.5, np.where(img_sticker==255,sticker_area,img_sticker).astype(np.uint8), 0.5, 0)
        else:
          img[y_min:y_max, x_min:x_max] = \
              np.where(img_sticker == 255, sticker_area, img_sticker).astype(np.uint8)
    
    return img
```
