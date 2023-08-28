# AIFFEL Campus Online 5th Code Peer Review
- 코더 : 최철웅
- 리뷰어 : 손정민


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [X] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  
- [x] 주석을 보고 작성자의 코드가 이해되었나요?
  > 어떤 과정인지 코드에 주석으로 표시되어 있었고, 잘 설명되어 있어 이해하기 쉬웠습니다.
  ```python
   def call(self, Q, K, V, mask):
        # Step 1: Linear_in(Q, K, V) -> WQ, WK, WV
        WQ = self.W_q(Q)
        WK = self.W_k(K)
        WV = self.W_v(V)
        
        # Step 2: Split Heads(WQ, WK, WV) -> WQ_split, WK_split, WV_split
        WQ_split = self.split_heads(WQ)
        WK_split = self.split_heads(WK)
        WV_split = self.split_heads(WV)

        # Step 3: Scaled Dot Product Attention(WQ_split, WK_split, WV_split)
        #         -> out, attention_weights
        out, attention_weights = self.scaled_dot_product_attention(WQ_split, WK_split, WV_split, mask)
        
        # Step 4: Combine Heads(out) -> out
        out = self.combine_heads(out)
        
        # Step 5: Linear_out(out) -> out
        out = self.linear(out)
  ```
- [x] 코드가 에러를 유발할 가능성이 없나요?
  > 에러 없이 잘 실행되었습니다.
- [x] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > 모델을 설계하고 훈련한 뒤 시각화까지 진행한 것으로 보아 코드를 이해하고 작성한 것 같습니다.
- [x] 코드가 간결한가요?
  > 시각화, 결과 출력 부분을 함수화했고 변수명 또한 직관적이었습니다. 코드 간 개행도 적절하여 가독성이 좋았습니다.
 ```python
     ids = []
    output = tf.expand_dims([tgt_tokenizer.bos_id()], 0)
    for i in range(dec_train.shape[-1]):
        # 마스크 생성
        enc_padding_mask, combined_mask, dec_padding_mask = \
            generate_masks(_input, output)

        # 다음 단어 추론
        predictions, enc_attns, dec_attns, dec_enc_attns =\
            model(_input, 
                  output,
                  enc_padding_mask,
                  combined_mask,
                  dec_padding_mask)
 ```
# 참고 링크 및 코드 개선
epoch마다 번역 결과가 출력되었다면 더 좋았을 것 같습니다. 이외에 개선할 부분은 보이지 않습니다.
