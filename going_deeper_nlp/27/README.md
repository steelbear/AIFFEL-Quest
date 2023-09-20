# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 최철웅
- 리뷰어 : 김성진


# PRT(Peer Review Template)
- [ ]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
    - 문제를 해결하는 완성된 코드란 프로젝트 루브릭 3개 중 2개, 
      퀘스트 문제 요구조건 등을 지칭
        - 해당 조건을 만족하는 코드를 캡쳐해 근거로 첨부
    
- [x]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
  주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드가 무슨 기능을 하는지, 왜 그렇게 짜여진건지, 작동 메커니즘이 뭔지 기술.
    - 주석을 보고 코드 이해가 잘 되었는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.

- 데이터셋 설명
```markdown
RM (Reward Model) 데이터셋
- prompt
    - 사용자가 올릴 질문
- completion_0, completion_1, completion_2
    - 각 모델별 답변
        - ChatGPT
        - GPT3(Ada)
        - GPT3(Davinci)
    - 어떤 답변이 어떤 모델로 생성됐는지 알수 없도록 답변 순서를 섞음
- ranking
    - 사람이 라벨링한 각 답변의 품질 랭킹
    - 값이 낮을수록 높은 품질
```

- 모델 디코딩
```python
# 예제 문장 토큰화
input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"].to(device)

# Beam Search Decoding
max_length = 128
output_beam = model.generate(input_ids, # 입력 시퀀스
                             max_length=max_length, # 생성 시퀀스 최대 길이
                             num_beams=7, # Beam Search할 범위
                             no_repeat_ngram_size=2, # 지정된 ngram 단위로 중복 체크
                             do_sample=True, # 토큰 샘플링
                             temperature=2.0, # 토큰 결정시 확률 반영도
                             top_k=50, # 후보 토큰 고를시 높은 확률순 k위까지만 보고 결정
                            )

# beam search로 만들어진 문장중 하나 출력
print(tokenizer.decode(output_beam[0]))
```
  
- [x]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나” 
  ”새로운 시도 또는 추가 실험을 수행”해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 문제에서 요구하는 조건에 더해 추가적으로 수행한 나만의 시도, 
      실험이 기록되어 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.

- 각 단계에서 데이터셋과 LM 과의 관계에 대한 이미지를 삽입하여 이해를 높였습니다.
  - SFT, RM, PPO
  
- [x]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.

```markdown
회고
- RLHF를 더 자세히 알 수 있었음
    - 처음에는 모델의 모든 대답을 사람이 평가하는 것으로 생각함
    - 하지만 일부 데이터셋에서만 라벨링을 한 후에 Reward Model을 만드는 방식이 신박하게 느껴짐
- LLM에 와서는 이제 학습 환경에 대해 신경을 써야함
    - 이제는 하나의 모델을 Fine-Tuning하는 것 조차 여러개의 모델을 필요로 함
    - GPU 환경에서 어떻게 학습할지 결정하기 위해서는 모델 학습이나 추론은 GPU에게 어떻게 맡기는지에 대한 지식이 필요함
```

    
- [x]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 하드코딩을 하지않고 함수화, 모듈화가 가능한 부분은 함수를 만들거나 클래스로 짰는지
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화했는지
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.

- 각 섹션별로 나뉘었고, 주석이 작성되었습니다.
- 꼭 필요한 부분은 클래스, 함수화되어있습니다.
```python
# 답변 생성 함수
def generation(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(
        torch.cuda.current_device())
    outputs = actor.generate(input_ids,
                             max_length=250,
                             do_sample=True,
                             top_k=50,
                             top_p=0.95,
                             num_return_sequences=1)
    output = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)[0]
    print()
    print(output)
    return output
```

# 참고 링크 및 코드 개선
```
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```