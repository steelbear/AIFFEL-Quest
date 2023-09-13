# AIFFEL Campus Online Code Peer Review
- 코더 : 최철웅
- 리뷰어 : 김성진


# PRT(Peer Review Template)
- [x]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
    - 문제를 해결하는 완성된 코드란 프로젝트 루브릭 3개 중 2개, 
      퀘스트 문제 요구조건 등을 지칭
        - 해당 조건을 만족하는 코드를 캡쳐해 근거로 첨부
    - [x] MLM, NSP task의 특징이 잘 반영된 pretrain용 데이터셋 생성과정이 체계적으로 진행되었다.
        - mask, NSP pair, np.memmap 과정으로 데이터셋이 잘 생성되었습니다.
    - [x] 학습진행 과정 중에 MLM, NSP loss의 안정적인 감소가 확인되었다.
        - 주피터 노트북에는 첨부되지 않았지만, 코드의 진행과정으로 볼 때 무난히 결과가 나올 것으로 판단됩니다.
    - [ ] 학습된 모델 및 학습과정의 시각화 내역이 제출되었다.
        - history 수치 내역을 시각화만 해주시면 될 것 같습니다.
    
- [x]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
  주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드가 무슨 기능을 하는지, 왜 그렇게 짜여진건지, 작동 메커니즘이 뭔지 기술.
    - 주석을 보고 코드 이해가 잘 되었는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.

> NSP pair 생성함수 설명
```markdown
NSP pair 생성 함수  create_pretrain_instances 정의
- `trim_tokens`
    - 문장의 길이를 일정 길이(max_seq)에 맞추는 함수
- `create_pretrain_instances`
    - 여러 문장이 담긴 doc을 받아 MLM과 NSP처리
    - 반환 타입
        - tokens
            - MLM과 NSP 처리된 시퀀스
        - segment
            - segment embedding에 전달할 세그먼트 구분 정보
            - 문장 A와 문장 B이 어디까지인지 표시
        - is_next
            - NSP 타겟 데이터
            - 문장 연결이 매끄러운지의 여부
                - 1: True(isNext)
                - 2: False(notNext)
        - mask_idx
            - MLM 타겟 데이터
            - 마스킹된 토큰의 인덱스 리스트
        - mask_label
            - MLM 타겟 데이터
            - 마스킹된 토큰의 정답 토큰 리스트
```


> NSP 헤더 설명
```markdown
NSP 헤더
- NSP 과제 해결을 위한 추가적인 헤더
- 출력 노드가 2개
  - MLM처럼 SparseCategoricalCrossentropy로 손실값을 계산하기 위함
  - binary classification을 class가 2개인 multi-class classification처럼 바꾼 것
```

- [x]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나” 
  ”새로운 시도 또는 추가 실험을 수행”해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 문제에서 요구하는 조건에 더해 추가적으로 수행한 나만의 시도, 
      실험이 기록되어 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.

> 위에 첨부한 내용과 같이 NSP 대해 이해하는 과정이 있었습니다.
  
- [x]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.

> 회고가 아래와 같이 잘 작성되었습니다.
> pre-trained 모델을 만들기 위한 방법에 대한 탐색과 모델 생성에 대한 생각이 담겨있습니다.

```markdown
회고
- pre-trained 모델을 만들기 위한 방법이 매우 방대하다
    - pre-train 이후 specific task를 학습하기 위한 헤더를 붙일 수 있는 모델 구조
    - 어떤 specific task든 빠르고 좋은 성능으로 학습할 수 있도록 pre-train하는 방법
    - 어떤 task도 모델이 받아내도록 도와주는 입력 데이터 template
    - 다양한 task를 가진 데이터셋
- pre-trained 모델은 만들어진 걸 쓰는것이 답이다
    - 그나마 작은 모델임에도 한 epoch에 20분
    - [Gugugo](https://github.com/jwj7140/Gugugo) 개발자가 왜 1epoch 학습후 결과를 올렸는지 이해하게됨
```
    
- [x]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 하드코딩을 하지않고 함수화, 모듈화가 가능한 부분은 함수를 만들거나 클래스로 짰는지
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화했는지
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.

> 함수화, 모듈화되어 간결하게 작성되었습니다.
```python
def create_pretrain_instances(vocab, doc, n_seq, mask_prob, vocab_list):
    """
    doc별 pretrain 데이터 생성
    :param vocab: SentencePieceProcess(), 토큰화 객체
    :param doc: MLM과 NSP 데이터로 만들 document, 각 문장마다 토큰화된 sentence 리스트
    :param n_seq: 최종 시퀀스의 최대 길이
    :param mask_prob: 각 토큰의 마스킹 확률
    :vocab_list: 특수 토큰을 제외한 사전 내 토큰 리스트
    :return instances: MLM과 NSP 처리된 학습 데이터셋, 위의 markdown 참조
    """
    # 특수 토큰 [CLS]과 [SEP]는 필수이므로 길이 제한에서 제외
    max_seq = n_seq - 3

    instances = []
    current_chunk = []
    current_length = 0
    for i in range(len(doc)):
        current_chunk.append(doc[i])  # line 단위로 추가
        current_length += len(doc[i])  # current_chunk의 token 수
        
        # 마지막 줄 이거나 길이가 max_seq 이상 인 경우
        if 1 < len(current_chunk) and (i == len(doc) - 1 or current_length >= max_seq):  
            # tokens_a와 tokens_b로 분리하기 위해 경계 고르기
            a_end = 1
            if 1 < len(current_chunk):
                a_end = random.randrange(1, len(current_chunk))
# (...이하 생략)
```

# 참고 링크 및 코드 개선
```
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```