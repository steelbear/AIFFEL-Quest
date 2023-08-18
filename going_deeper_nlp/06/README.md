# AIFFEL Campus Online 5th Code Peer Review
- 코더 : 최철웅
- 리뷰어 : 김성진


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [X] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  > 저의 환경이 같지 않아서 직접 실행은 하지 못했지만, 코드 상으로는 문제를 해결한 것으로 보입니다.
- [X] 주석을 보고 작성자의 코드가 이해되었나요?
  > 위 항목에 대한 근거 작성 필수
  > 네, 작업에 따른 섹션별로 나뉘어 있고,
  > 예상, 확인 결과, 작업할 내용 등이 잘 작성되어 있습니다.
- [X] 코드가 에러를 유발할 가능성이 없나요?
  > 위 항목에 대한 근거 작성 필수
  > 네, 에러를 유발할 가능성을 찾지 못했습니다.
- [X] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 위 항목에 대한 근거 작성 필수
  > 네, 코드 흐름, 함수화, 루프 등을 볼 때 그렇습니다.
- [X] 코드가 간결한가요?
  > 위 항목에 대한 근거 작성 필수
  > 네, 간결합니다.

# 예시
1. 코드의 작동 방식을 주석으로 기록합니다.
2. 코드의 작동 방식에 대한 개선 방법을 주석으로 기록합니다.
3. 참고한 링크 및 ChatGPT 프롬프트 명령어가 있다면 주석으로 남겨주세요.

> 작업할 내용 작성

```markdown
- vocabulary size에 따른 모델 성능 평가
    1. vocabulary size 정하기
    2. 데이터 불러오기
    3. 모델 훈련
    4. 모델 평가
- 평가해볼 vocabulary size 정하기
    - 문제에서 정해진 크기
        - None, 5000
    - 전체에서 <unk> 처리되는 비율에 따라 나누기 
```

> 예상 및 확인 내용 작성

```markdown
테스트할 vocabulary size
- 119: 50%
- 773: 75%
    - 단어수가 극단적으로 작은 경우 모델이 잘 판단할 수 있는지 확인해보기
- 3166: 90%
- 5000: 93.4%
- 6380: 95%
- 106218: 97.5%
- 30982(None): 100%

백분위수에 해당하는 단어 확인하기
- 예상
    - 인덱스 번호가 커질수록 고유명사가 많이 등장할 것
- 확인 결과
    - 백분위수를 중심으로 10개 단어 확인
    - 50%를 제외하면 별 차이는 없어보임
    - 75% 이후로는 고유명사 빈도가 비슷한 것으로 보임
```

> 핵심 모델 훈련 루프

```python
# 주어진 vocabulary size로 모델 학습 후 평가
def train_and_evaluate(num_words):
    # 정해진 크기만큼 어휘 사전 생성
    (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=num_words, test_split=0.2)
    x_train, x_test = vectorize_ducuments(x_train, x_test)
    
    models = [
        MultinomialNB(), ComplementNB(),
        DecisionTreeClassifier(max_depth=10), 
        RandomForestClassifier(n_estimators=5, max_depth=10, n_jobs=8),
        LogisticRegression(C=10000, penalty='l2', max_iter=3000, n_jobs=8), 
        LinearSVC(C=1000, penalty='l1', max_iter=3000, dual=False),
        GradientBoostingClassifier(),
    ]
    
    # VotingClassifier 추가
    estimators = [('lr', LogisticRegression(C=10000, penalty='l2', max_iter=3000)), 
                  ('rf', RandomForestClassifier(n_estimators=5, max_depth=10)), 
                  ('gnb', GradientBoostingClassifier()),
                 ]
    models.append(VotingClassifier(estimators=estimators, voting='soft', n_jobs=8))
    
    # 정확도와 f1 score를 모델마다 저장
    accuracies = {}
    f1_scores = {}
    for model in tqdm(models):
        # 학습
        model.fit(x_train, y_train)
        # 평가
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = np.zeros((46,))
        for i in range(46):
            f1[i] = f1_score(y_test, y_pred, average='micro')
        accuracies.update({model.__class__.__name__: accuracy})
        f1_scores.update({model.__class__.__name__: f1})
        
    return accuracies, f1_scores```

> 단어 빈도, 백분위수 확인 내용이 인상적입니다. 감사합니다.


# 참고 링크 및 코드 개선
```python
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```
