# AIFFEL Campus Online 5th Code Peer Review
- 코더 : 최철웅
- 리뷰어 : 본인의 이름을 작성하세요.


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [X] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
- [X] 주석을 보고 작성자의 코드가 이해되었나요?
  > 코드가 주석과 마크다운으로 잘 설명되어 있어 이해하기 쉬웠습니다.
  ```python
  # 한국어와 영어 문장을 다시 분리
  for kor, eng in cleaned_corpus:
      # 학습 속도를 위해 40 단어를 초과하는 문장 제거
      if len(kor.split()) > 40 or len(eng.split()) > 40:
          continue
  ```
- [X] 코드가 에러를 유발할 가능성이 없나요?
  > 에러 없이 잘 실행되었습니다.
- [X] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 과제를 수행하고 Epoch 횟수에 따른 결과를 정리한 것, 주석의 내용으로 보아 코드를 잘 이해하고 있는 것 같습니다.
- [X] 코드가 간결한가요?
  > 적절한 개행과 직관적인 변수명을 사용하여 가독성이 좋았습니다.


# 참고 링크 및 코드 개선
매 Epoch마다 예시 문장의 결과를 출력하여 Epoch 수에 따른 변화를 알 수 있게 코드를 작성한 점이 인상깊었습니다. `batch size`나 `embed_dim`, `units`의 값을 변경해보는 것도 좋은 시도가 될 것 같습니다.