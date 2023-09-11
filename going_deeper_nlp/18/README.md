# AIFFEL Campus Online Code Peer Review
- 코더 : 최철웅
- 리뷰어 : 김석영


# PRT(Peer Review Template)
- [ ]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 챗봇 훈련데이터를 위한 전처리와 augmentation이 적절히 수행됨.
      ```python
        que_corpus_augmented = []
        ans_corpus_augmented = []
        
        for question, answer in tqdm(zip(que_corpus, ans_corpus), total=len(que_corpus)):
            # 어휘 대치
            question_augmented = lexical_sub(question, wv)
            answer_augmented = lexical_sub(answer, wv)
            
            # 원본, 원본 쌍
            que_corpus_augmented.append(question)
            ans_corpus_augmented.append(answer)
            
            # augmented, 원본 쌍
            que_corpus_augmented.append(question_augmented)
            ans_corpus_augmented.append(answer)
            
            # 원본, augmented 쌍
            que_corpus_augmented.append(question)
            ans_corpus_augmented.append(answer_augmented)
            
        print(len(que_corpus_augmented), len(ans_corpus_augmented))
      ```
    
- [ ]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
  주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 어휘를 대치하는 lexical_sub 함수의 기능을 단계별로 잘 구분/주석처리해 이해가능하게 구현함.
      ```python
        # 어휘 대치 함수
        def lexical_sub(sample_sentence, wv):
            sample_tokens = sample_sentence.split()
        
            # 랜덤으로 Word2Vec 사전에 존재하는 단어 고르기
            for _ in range(len(sample_tokens)):
                selected_tok = random.choice(sample_tokens)
                if selected_tok in wv.key_to_index:
                    break
                else:
                    # 못찾는다면 어휘 대치 하지 않음
                    selected_tok = None
        
            result = ""
            for tok in sample_tokens:
                if tok is selected_tok:
                    result += wv.most_similar(tok)[0][0] + " "
                else:
                    result += tok + " "
            return result
      ```
  
- [ ]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나” 
  ”새로운 시도 또는 추가 실험을 수행”해봤나요?**
    - 성능 측정 시, 모델 예측과 BLEU SCORE를 결합해 보기좋게 출력함.
      ```python
        def calculate_bleu(reference, candidate, weights=[0.25, 0.25, 0.25, 0.25]):
            return sentence_bleu([reference],
                                 candidate,
                                 weights=weights,
                                 smoothing_function=SmoothingFunction().method1)

        enc_samples = enc_train[0:15:3, :]
        dec_samples = dec_train[0:15:3, :]
        
        for i in range(len(test_text)):
            question = vectorizer.decode_ids(enc_samples[i].tolist())
            answer = vectorizer.decode_ids(dec_samples[i].tolist())[9:]
            print("질문:", question)
            print("챗봇 대답:", dec_preds[i])
            print("원본 대답:", answer)
            print("BLEU score:", calculate_bleu(answer, dec_preds[i]))
      ```
  
- [ ]  **4. 회고를 잘 작성했나요?**
    - SentencePiece에서 bpe 방식으로 토큰화할 때 vocab_size가 최대 개수를 넘어가면 무한 루프에 빠짐
      + 지금까지 bpe로 돌렸을때 4시간동안 안되는 이유
      + unigram의 경우에는 Warning 문구로 가능한 최대 단어 개수를 초과했다고 알려줌
    - SentencePiece 라이브러리를 좀 더 살펴봐야함
      + <start>와 <end>를 토큰으로 등록해도 인코딩 문제가 존재함
    
- [ ]  **5. 코드가 간결하고 효율적인가요?**
    - 데이터 전처리와 토큰화 작업을 함수화함.
    ```python
    def preprocess_sentence(sentence):
        sentence = sentence.lower()
        for p in ['!', '?', '.', ',']:
            sentence = sentence.replace(p, ' ' + p + ' ')
        sentence = re.sub(r'[^a-zA-Zㄱ-ㅎ가-힣0-9!?\., ]', ' ', sentence)
        sentence = re.sub(r' +', ' ', sentence)
        sentence = sentence.strip()
        
        return sentence

    mecab = Mecab()

    def build_corpus(sentences, tokenizer):
        corpus = []
        for sentence in sentences:
            sentence = preprocess_sentence(sentence)
            tokens = tokenizer.morphs(sentence)
            corpus.append(' '.join(tokens))
        return corpus
    ```

# 참고 링크 및 코드 개선
```

```
