# AIFFEL Campus Online 5th Code Peer Review
- 코더 : 최철웅
- 리뷰어 : [이태훈](https://github.com/git-ThLee)


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [X] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  > 네! 사전학습된 Word2Vec를 사용한 실습(과제)은 시간만 충분했으면 수행했을 것 같습니다. 이미 코드는 다 짜어져 있습니다.

- [X] 주석을 보고 작성자의 코드가 이해되었나요?
  > 네! 최대 문장 길이 값이 내 코드와 달라서 주석으로 빠르게 찾을 수 있습니다.

```python
import math

# 평균과 표준편차로 최대 문장 길이 계산
max_length = math.floor(length_mean + 3 * length_std)

# 최대 문장 길이보다 짧은 문장 비율 확인하기
count_shorter_sentences = len(list(filter(lambda x: x < max_length, sentence_lengths)))

print("문장의 최대 길이를 {}로 잡으면".format(max_length))
print("{:.2f}%의 문장들이 잘리지 않습니다.".format(count_shorter_sentences / len(sentence_lengths) * 100))
```

- [X] 코드가 에러를 유발할 가능성이 없나요?
  > 네! 체계적으로 작성되어 있어서, 다양한 실험에 필요한 파라미터 선택도 빠르게 가능합니다.

```python
from tensorflow.keras.callbacks import EarlyStopping

# 학습에 공통된 하이퍼 파라미터 적용
fit_options = {
    'epochs': 10,                      # 훈련 횟수
    'batch_size': 512,                 # 배치 사이즈
    'callbacks': [EarlyStopping(patience=3)],    # 조기 종료 (검증 데이터셋에서 성능 향상이 되지 않았다면 훈련 종료)
    'validation_data': (X_val, y_val), # 검증 데이터셋
    'verbose': 1,                      # 모델 훈련 정보 표시
    'workers': 4                       # 프로세스 수
}

# 모델 학습
print("Training SimpleRNN ...")
history_rnn = model_rnn.fit(X_train, y_train, **fit_options)
print("Training LSTM ...")
history_lstm = model_lstm.fit(X_train, y_train, **fit_options)
print("Training GRU ...")
history_gru = model_gru.fit(X_train, y_train, **fit_options)
```

- [X] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 네! 제공해주는 BaseCode를 사용하지 않고, 직접 코드를 구현하셨습니다!

```python
def load_data(train_data, test_data, num_words=num_words):
    # 결측치 제거
    train_data.dropna(inplace=True)
    test_data.dropna(inplace=True)
    
    # 중복 데이터 제거
    train_data = train_data.drop_duplicates(subset=['document'])
    test_data = test_data.drop_duplicates(subset=['document'])
    
    # 데이터셋 내 모든 형태소 수집
    all_word_list = []
    for _, row in tqdm(train_data.iterrows(), total=train_data.shape[0]):
        word_list = tokenizer.morphs(row['document'])
        word_list = [word for word in word_list if word not in stopwords]
        all_word_list.extend(word_list)
    
    for _, row in tqdm(test_data.iterrows(), total=test_data.shape[0]):
        word_list = tokenizer.morphs(row['document'])
        word_list = [word for word in word_list if word not in stopwords]
        all_word_list.extend(word_list)
        
    # 단어와 정수를 맵핑
    counter = Counter(all_word_list)
    word_to_index = {k:(v+3) for v, (k, _) in enumerate(sorted(counter.items(), key=lambda x: x[1], reverse=True))}
    word_to_index['<PAD>'] = 0    # 패딩용 토큰
    word_to_index['<BOS>'] = 1    # 문장 시작 토큰
    word_to_index['<UNK>'] = 2    # 사전에 없는 단어 토큰
    word_to_index['<UNUSED>'] = 3 # 사전에서 짤린 단어 토큰
    
    # 빈도를 기준으로 num_words개의 단어만 고유 토큰 부여
    # 빈도가 많은 단어만 학습시키기
    if num_words != -1:
        for k, v in word_to_index.items():
            if v > num_words:
                word_to_index[k] = 3
    
    # 훈련 데이터셋을 입력 데이터와 정답 데이터로 분리
    x_train = train_data['document'].values
    y_train = train_data['label'].values
    
    # 테스트 데이터셋을 입력 데이터와 정답 데이터로 분리
    x_test = test_data['document'].values
    y_test = test_data['label'].values
    
    return x_train, y_train, x_test, y_test, word_to_index
```

- [X] 코드가 간결한가요?
  > 네! 

```python
import matplotlib.pyplot as plt
import seaborn as sns


# 각 문장의 길이를 측정
X_train_length = list(map(len, X_train_tokenized))
X_test_length = list(map(len, X_test_tokenized))

# 학습 데이터셋과 테스트 데이터셋의 문장 길이 분포 그래프 그리기
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

sns.countplot(X_train_length, ax=axes[0], hue=y_train)
sns.countplot(X_test_length, ax=axes[1], hue=y_test)

axes[0].set_title('train')
axes[1].set_title('test')


plt.show()
```



# 참고 링크 및 코드 개선
```python
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```
