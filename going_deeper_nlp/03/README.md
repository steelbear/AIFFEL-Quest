# AIFFEL Campus Online 5th Code Peer Review
- 코더 : 최철웅
- 리뷰어 : 김석영


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [X] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  > 네. 프로젝트에서 제시한 네 가지 STEP과 평가문항의 기준들을 모두 이행하였습니다.
- [X] 주석을 보고 작성자의 코드가 이해되었나요?
  > 네. Task별로 제목과 코드가 잘 구분이 돼 있고, 시각화가 잘 돼 있어 이해하기 용이했습니다.
- [X] 코드가 에러를 유발할 가능성이 없나요?
  > 네. 에러 유발 인자는 찾아보기 힘딥니다.
- [X] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 네. Task 자체가 누적/비교하는 부분이 많았지만 에러 없이 잘 구현했고, 또 전반적으로 코드가 세부적인 부분까지 구현이 되었어서 충분히 이해를 바탕으로 작성된 코드라 할 수 있습니다.
- [X] 코드가 간결한가요?
  > 네. 필요한 코드들 위주로 작성이 되었고, 반복된 코드의 비율도 낮은 편이라 간결하게 작성됐다고 평가할 수 있습니다.

# 예시
1. 코드의 작동 방식을 주석으로 기록합니다.
2. 코드의 작동 방식에 대한 개선 방법을 주석으로 기록합니다.
3. 참고한 링크 및 ChatGPT 프롬프트 명령어가 있다면 주석으로 남겨주세요.
```python
def sp_tokenizer(s, corpus, labels, padding='post', show_freq=True):
    sequences = []
    targets = []

    for i, text in enumerate(corpus):
        sequence = s.EncodeAsIds(text)
        if len(sequence) < 5:
            continue
        sequences.append(sequence)
        targets.append(labels[i])
    
    if show_freq:
        seq_lengths = list(map(len, sequences))
        _, ax = plt.subplots(figsize=(10, 20))
        sns.countplot(y=seq_lengths, ax=ax)
        ax.set_ylabel('length of sequence')
        ax.set_xlabel('count')
        plt.show()
    
    sequences = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=70, padding=padding)
    return sequences, np.array(targets)

train_inputs, train_labels = sp_tokenizer(s, _corpus, _labels)
```

# 참고 링크 및 코드 개선
전반적으로 잘 작성된 코드이기도 하고, 또 코드 리뷰 peer가 바뀌어 20분 간 리뷰 작성만 하므로, 
코드 개선 관련 코멘트할 수 있는 시간의 제한이 있으니 이를 참조해주시기 바랍니다.
