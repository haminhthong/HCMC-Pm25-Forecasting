# Model Card — Next-Hour PM2.5 Forecasting Platform for HCMC

## 1. Mục Đích & Phạm Vi Nghiệp Vụ (Model Details)

- **Bài toán**: Dự báo nồng độ ô nhiễm bụi mịn PM2.5 giờ tiếp theo ($t \rightarrow t+1$) theo từng trạm quan trắc đơn lẻ tại TP.HCM.
- **Mục tiêu kỹ thuật**: Cung cấp dự báo điểm kèm prediction interval 90% được hiệu chuẩn trên future calibration window, tích hợp Quality Gate fallback về Persistence Baseline khi mô hình không vượt baseline. Coverage được monitor theo block thời gian, không claim bảo đảm vô điều kiện cho time series.
- **Phạm vi sử dụng**: Artifact phục vụ trình diễn kỹ thuật Machine Learning Engineering, MLOps, dự báo chuỗi thời gian chống data leakage; không thay thế cho hệ thống cảnh báo sức khỏe môi trường chính thức của cơ quan nhà nước.

---

## 2. Dữ Liệu Đầu Vào & Đầu Ra (Inputs & Outputs)

### 2.1 Đầu vào (Input Schema per Single-Station Request):
- `timestamp`: Chuỗi thời gian chuẩn ISO 8601, tăng dần và liên tục theo từng giờ ($\Delta t = 1\text{h}$).
- `station_id`: Tên định danh trạm quan trắc (duy nhất 1 trạm/request).
- `PM2.5`: Nồng độ PM2.5 tại mốc thời điểm hiện tại $t$ ($\ge 0 \;\mu\text{g/m}^3$).
- **Lịch sử tối thiểu**: 25 quan trắc giờ liên tục để phục vụ tạo lag 24h và rolling window 24h.
- **Biến ngoại sinh tùy chọn**: $O_3$, $SO_2$, $NO_2$, $CO$, $TSP$, nhiệt độ, độ ẩm (nếu thiếu, pipeline tự động điền median bằng SimpleImputer).

### 2.2 Đầu ra (Standard Output Schema):
- `station_id`: Tên trạm quan trắc.
- `forecast_origin`: Mốc thời điểm hiện tại $t$ của quan trắc.
- `forecast_for`: Mốc thời điểm $t+1\text{h}$ được dự báo.
- `current_pm25`: Nồng độ PM2.5 hiện tại tại $t$.
- `predicted_pm25`: Nồng độ PM2.5 dự báo tại $t+1$.
- `level`: Mức phân tích nội bộ (`Thấp` $\le 12.0$, `Trung bình` $< 35.5$, `Cao` $\ge 35.5$).
- `forecast_strategy`: Chiến lược suy luận (`"ml_model"` hoặc `"persistence_fallback"`).
- `serving_champion`: Tên mô hình chính thức phục vụ suy luận (`ridge`, `persistence`, ...).
- `interval`:
  - `method`: `"split_conformal"`
  - `coverage_target`: `0.9` (90% target coverage)
  - `lower`: Cận dưới prediction interval ($\ge 0.0$).
  - `upper`: Cận trên prediction interval.
  - `width`: Độ rộng prediction interval ($2 \times q_{90}$).
- `model_version`: Mã định danh phiên bản tự động (`pm25-YYYYMMDD-<git_sha>-<data_hash>`).
- `updated_at`: Thời điểm hệ thống sinh dự báo.

---

## 3. Kiến Trúc Pipeline 7 Giai Đoạn (Canonical 7-Stage Pipeline)

1. **Data Ingestion & Audit**: Kiểm tra schema, tính đơn điệu của timestamp, phát hiện trùng lặp hoặc khoảng trống giờ bất thường.
2. **Time-Aware Feature Engineering**:
   - Clock-time lag: tra cứu theo $(station\_id, timestamp - lag)$, không dùng shift theo vị trí dòng.
   - Rolling statistics (mean, std): sử dụng `closed="left"` để chỉ tính lịch sử trước $t$.
   - Trend deltas: chênh lệch $y_t - y_{t-1\text{h}}$ và $y_t - y_{t-3\text{h}}$.
   - Cyclic time encoding: $\sin/\cos(2\pi \cdot \text{hour}/24)$, day of week.
3. **Nested Temporal Partition**:
   - Chia thành Train, Independent Calibration và Final Test sao cho `target_timestamp < cal_start` và `target_timestamp < test_start`.
   - Trong tập Train: thiết lập Expanding-Window Cross-Validation Folds với `target_timestamp < validation_start`.
4. **Model Selection**:
   - Đánh giá các baselines: **Persistence** ($\hat{y}_{t+1} = y_t$), **Seasonal Naive 24h** ($\hat{y}_{t+1} = y_{t-23}$), **Ridge Autoregression**.
   - Đánh giá các mô hình ensemble: **Random Forest**, **ExtraTrees**, **HistGradientBoosting**.
   - Chọn Candidate Champion theo tiêu chí $\text{Mean CV MAE}$ thấp nhất.
5. **Calibration & Quality Gate**:
   - Huấn luyện Candidate Champion trên toàn bộ tập Train.
   - Đánh giá trên tập Calibration độc lập để thu thập phân phối phần dư cho cả Candidate ML và Persistence.
   - Tính quantile bậc 90% ($q_{90}$) cho Conformal Prediction Interval.
   - Thẩm định Quality Gate đa tiêu chí:
     1. $MAE_{\text{model}} \le MAE_{\text{persistence}} \times (1 - 0.05)$
     2. $\text{Recall}_{\text{Cao}} \ge 0.75$
     3. $\text{Std}(MAE_{\text{folds}}) \le 1.0$
   - Nếu Quality Gate **PASS**: `serving_champion` = Candidate ML.
   - Nếu Quality Gate **FAIL**: `serving_champion` = `persistence` (fallback an toàn).
6. **Freeze Policy & Final Test**:
   - Khóa chính sách suy luận trước khi đánh giá tập test cuối.
   - Báo cáo riêng biệt: `candidate_ml_test`, `persistence_test`, `seasonal_naive_test`, và `serving_champion_test`.
   - Tính toán MASE và MAE Skill Score so với Persistence.
   - Đo lường độ phủ thực tế (PICP) và độ rộng trung bình (MPIW) của Conformal Interval trên tập Test cuối.
   - Thực hiện phân tích sai số phân rã (Sliced Error Analysis) theo Trạm, Giờ trong ngày, và Mức độ ô nhiễm.
7. **Serving & Monitoring**:
   - Đóng gói artifact: `model.joblib`, `metadata.json`, `evaluation.json`, `feature_schema.json`, `config_snapshot.yaml`.
   - Phục vụ trực tiếp qua FastAPI và giám sát trực quan qua Streamlit Dashboard.

---

## 4. Chỉ Số Đánh Giá Chuẩn Mực

- **Chỉ số Hồi quy chính (Headline Metrics)**:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - Bias ($\text{mean}(\hat{y} - y)$)
  - P90 Absolute Error
  - MASE ($\frac{MAE_{\text{model}}}{MAE_{\text{persistence}}}$)
  - Skill Score vs Persistence ($1 - \frac{MAE_{\text{model}}}{MAE_{\text{persistence}}}$)
- **Chỉ số Phân rã & Diễn giải (Downstream Operational Interpretation)**:
  - Macro-F1 score
  - Quadratic Weighted Kappa (QWK)
  - High PM2.5 Recall ($\ge 35.5 \;\mu\text{g/m}^3$)
  - Confusion Matrix
- **Chỉ số Độ tin cậy (Uncertainty Reliability)**:
  - PICP (Prediction Interval Coverage Probability, mục tiêu 90%)
  - MPIW (Mean Prediction Interval Width)
  - Median Interval Width

---

## 5. Giới Hạn & Hướng Phát Triển

- **Dữ liệu mẫu**: Tập dữ liệu hiện tại là tập dữ liệu mẫu phục vụ kiểm định kiến trúc. Cần tích hợp nguồn dữ liệu quan trắc thực tế (OpenAQ, đài quan trắc khí hậu) cho bài toán sản xuất.
- **Phạm vi Conformal**: Hiện tại áp dụng Marginal Split Conformal. Kế hoạch tiếp theo là nâng cấp lên Per-station Conformal và Low/Medium/High Regime Calibration.
- **Horizon**: Hỗ trợ mở rộng từ $t+1$ sang Multi-horizon ($t+1, t+3, t+6, t+12, t+24\text{h}$) với Direct Forecasting Models.
