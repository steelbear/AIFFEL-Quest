# AIFFEL Campus Online Code Peer Review
- 코더 : 최철웅
- 리뷰어 : 김석영


# PRT(Peer Review Template)
- [X]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 네. 루브릭의 평가문항과 상세기준을 충족하는 완성된 코드가 제출되었습니다.
    
- [X]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
  주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 네. Dynamic Padding 추가 후 작성한 training_arguments와 Trainer 코드에 주석이 잘 기재됐고, 코드 또한 잘 이해됐습니다.
     
    '''python

    training_arguments = TrainingArguments(
        output_dir,
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        #per_device_train_batch_size=4, # Dynamic Padding으로 더 큰 batch로 돌릴 수 있도록
        #per_device_eval_batch_size=4,  # 직접 지정한 batch_size를 해제
        num_train_epochs=1,
        weight_decay=0.01,
        group_by_length=True # 길이가 비슷한 데이터끼리 묶기
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator # data_collator 추가
    )

    '''
  
- [X]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나” 
  ”새로운 시도 또는 추가 실험을 수행”해봤나요?**
    - 네. 에러가 난 부분은 없었으며, fine tuning으로 성능을 개선을 시도한 코드와 수행 결과가 있습니다.

    '''python

    # 가중치를 저장할 위치
    output_dir = os.getenv('HOME') + '/aiffel/transformers'
    
    training_arguments = TrainingArguments(
        output_dir,
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        weight_decay=0.01,
    )
    training_arguments

    # fine-tuning을 위한 trainer
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # 모델 fine-tuning
    trainer.train()
    
    '''
  
- [X]  **4. 회고를 잘 작성했나요?**
    - 네. 회고도 잘 작성됐습니다. (코드 하단 부분 참고)
    
- [X]  **5. 코드가 간결하고 효율적인가요?**
    - 네. 전체적으로 코드들이 간결하게 작성돼 있습니다. 실제 코드 길이가 짧고 간단합니다.
  
    '''python

    # 전처리시 패딩 미적용
    def transform_without_padding(data):
        return tokenizer(
            data['document'],
            truncation=True,
            return_token_type_ids=False,
        )

    '''

# 참고 링크 및 코드 개선
```

```
