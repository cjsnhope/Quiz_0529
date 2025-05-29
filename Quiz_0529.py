print(" HR 데이터 기반 이직 예측 실습 ")

# 라이브러리 호출
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# 1. 데이터 로드
file_path = "한국_기업문화_HR_데이터셋_샘플.csv"
# 2. 파일 불러오기
df = pd.read_csv(file_path)

# 3. 데이터 전처리
# 3-1. 결측치 처리
assert df.isnull().sum().sum() == 0, "결측치 있음"
# 3-2. 이직여부 이진값 변환
df["이직여부"] = df["이직여부"].map({"No": 0, "Yes": 1})
# 3-3. 범주형 인코딩 (LabelEncoder 적용)
categorical_cols = df.select_dtypes(include="object").columns.tolist()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 4. 피처 선택
# 5. 모델 훈련
selected_features = ["Age", "출장빈도", "부서", "야근여부", "업무만족도", "집까지거리", "근무환경만족도", "워라밸"]
# 선택 근거: Age(젊을수록 이직), 출장빈도(많을수록 피로->이직), 부서(부서별 이직률 차이O), 야근여부(많을수록 스트레스->이직), 업무만족도(낮을수록 이직), 집까지거리(멀수록 이직), 근무환경만족도(낮을수록 이직), 워라밸(낮을수록 이직)
X = df[selected_features]
y = df["이직여부"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 8:2 분할
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 6. 성능 검증
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("정확도:", accuracy)
print("혼동행렬:\n", conf_matrix)

