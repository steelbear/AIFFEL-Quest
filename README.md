# AIFFEL Quest
아이펠 캠퍼스에서 진행하는 퀘스트 실습 결과물을 올리는 레포지토리입니다.

## Exploration
| 차시 | 제목 | 내용 |
|----|-----------------------------------------|----------|
| 1  | 당뇨병 수치 예측하기, 자전거 수요 예측하기 | Numpy로 간단한 Linear Regression 모델과 학습 루프를 구현하고 당뇨병 수치 예측, Scikit-Learn을 사용하여 자전거 수요 예측 |
| 2  | 주택 가격 예측하기                       | Pandas, MissingNo, Scikit-learn을 통한 데이터 전처리, Matplotlib와 Seaborn을 통한 데이터 시각화, 트리 기반 모델을 통한 주택가격 예측 |
| 3  | Facial Detection                        | 컴퓨터 비전에서 Matplotlib를 다루기, dlib를 이용한 얼굴 인식 체험 |
| 4  | 영화 리뷰의 긍정/부정 분류하기            | 텍스트 전처리 및 토큰화, 벡터화, RNN 기반 모델을 통한 이진 감정분석 |
| 5  | Segmentation을 이용한 사진 필터          | OpenCV를 이용한 간단한 이미지 프로세싱, 세그멘테이션 모델을 통해 인물 사진에 필터 적용하기 |
| 6  | 뉴스기사 요약봇 만들기                    | Seq2Seq 모델을 통한 추상적 요약, summa 라이브러리를 통해 추출적 요약 체험 |
| 7  | Segmentation Map으로 도로 이미지 생성하기 | Conditional GAN을 통해 Segmentation 이미지를 실제 도로 이미지로 변환 |
| 8  | 한국어 데이터로 챗봇 만들기               | Transformer를 직접 구현하고 한국어 데이터를 학습시키기 |

## Going Deeper (NLP)
| 차시 | 제목 | 내용 |
|----|-----------------------------------|----------|
| 3  | 영화 리뷰 감정 이진분류             | 기초적인 텍스트 전처리, KoNLPy와 SentencePiece을 이용한 토큰화, LSTM을 이용한 감정분석 |
| 6  | 뉴스 카테고리 다중분류              | Scikit-learn을 통해 고전적 머신러닝 모델로 텍스트를 분류 |
| 9  | Word Embedding의 편향성 평가       | 명사만 추출해 학습한 Word2Vec 모델을 WEAT로 데이터 편향성 평가 |
| 12 | 한국어 - 영어 기계번역             | Bahdanau Attention을 사용한 Seq2Seq 모델에 한영 기계번역 학습 |
| 15 | Transformer를 이용한 기계번역      | Tensorflow와 Keras로 직접 Transformer 모델을 구현하고 기계번역 학습 |
| 18 | Transformer를 이용한 챗봇 제작     | Data Augmentation을 통해 챗봇의 대답 품질 향상시키기 |
| 21 | BERT 구현 및 Pre-Training         | Tensorflow와 Keras로 직접 BERT를 구현하고 Pre-Training 수행 |
| 24 | HuggingFaces을 활용한 Fine-Tuning | HuggingFaces의 Transformers를 이용해 KLUE-BERT를 Fine-Tuning 후 감정분석 수행 |
| 27 | RLHF을 통해 ChatGPT 만들기        | KoChatGPT를 통해 RLFH을 수행해 모델의 답변 품질 향상시키기 |

## SlowPaper
- [LLM.int8()](https://arxiv.org/abs/2208.07339) 논문을 읽고 정리한 내용을 발표
