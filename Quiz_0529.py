
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
# 결과 해석: 정확도는 0.85이고, 혼동행렬을 봤을 때 이직하는 안 하는 직원은 잘 예측했지만 이직하는 직원은 다수 틀렸음.

# 7. 예측 결과 분석
y_proba = model.predict_proba(X_test)[:, 1]
X_test_with_proba = X_test.copy()
X_test_with_proba["이직확률"] = y_proba
top5_risk = X_test_with_proba.sort_values(by="이직확률", ascending=False).head(5)
print("이직 예측 직원 수:", (y_pred == 1).sum())
print("상위 이직 위험 직원 5명:\n", top5_risk) 

# 8. 신입사원 예측
data = [
    {"Age": 29, "출장빈도": 2, "부서": 1, "야근여부": 1, "업무만족도": 2, "집까지거리": 5, "근무환경만족도": 2, "워라밸": 2},
    {"Age": 42, "출장빈도": 0, "부서": 0, "야근여부": 0, "업무만족도": 4, "집까지거리": 10, "근무환경만족도": 3, "워라밸": 3},
    {"Age": 35, "출장빈도": 1, "부서": 2, "야근여부": 1, "업무만족도": 1, "집까지거리": 2, "근무환경만족도": 1, "워라밸": 2},
]
new_df = pd.DataFrame(data)
new_preds = model.predict(new_df)
print("신입사원 이직 예측:", new_preds.tolist())

# 9. 이직 영향 컬럼 파악 및 결과 해석
importances = model.feature_importances_
features = X_train.columns
importance = pd.Series(importances, index=features).sort_values(ascending=False)
# 상위 3개 피처 출력
top3_features = importance.head(3)
print("피처 중요도 상위 3개:")
print(top3_features)
# Age: 젊을수록 안정성이 낮고 이직하려는 경향 높음, 집까지거리: 거리가 멀고 출퇴근 시간 오래 걸릴 수록 이직 가능성이 높아짐, 업무만족도: 업무에 대한 성취감과 만족감이 낮을 수록 다른 직무나 회사를 찾아 이직함

