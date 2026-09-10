# Model card

## Mục tiêu

Prototype dự báo PM2.5 của một trạm cho giờ kế tiếp từ quan trắc lịch sử theo giờ. Dữ liệu mặc định là `data/sample/air_quality_sample.csv`.

## Quy trình

1. Chuẩn hóa timestamp và `available_at` về UTC.
2. Regularize chuỗi theo lưới một giờ, giữ các mốc thiếu dưới dạng `NaN`.
3. Tạo clock-time lag, rolling causal, trend, time và exogenous features.
4. Chia theo thời gian thành train, calibration và final test.
5. Chọn `best_cv_model` bằng expanding-window backtest.
6. So sánh model với Persistence trong calibration để chọn `forecast_strategy`.
7. Tạo conformal interval từ residual của calibration window riêng.
8. Nếu history runtime thiếu hoặc có gap vượt chính sách, dùng Persistence.

## Artifact và inference

Artifact hiện tại gồm `model.joblib`, `metadata.json`, `evaluation.json` và `feature_schema.json`. Cấu hình phục vụ được đọc từ `configs/config.yaml`; không có registry hoặc release pointer.

Metadata ghi:

- `best_cv_model`: model có MAE CV thấp nhất;
- `forecast_strategy`: tên model được dùng hoặc `persistence`;
- `dataset_scope`: `sample` hoặc `external`;
- feature columns, input policy, conformal interval và data provenance.

## Baseline và metric

Persistence là baseline bắt buộc cho dự báo `y(t+1) = y(t)`. Seasonal Naive 24h là baseline bổ sung. Báo cáo gồm MAE, RMSE, MASE, skill score, macro-F1, recall nhóm PM2.5 cao, PICP và độ rộng interval.

## Hạn chế

- Sample rất nhỏ, final test hiện chỉ khoảng 8 dòng và calibration khoảng 6 dòng.
- Kết quả sample không đại diện cho toàn TP.HCM.
- Interval chỉ minh họa triển khai split-conformal; không được diễn giải là calibration đáng tin cậy trên dữ liệu sản xuất.
- Các ngưỡng `Thấp/Trung bình/Cao` là nhãn phân tích nội bộ, không phải AQI chính thức.
- Chưa có connector station API, lưu trữ incremental, xử lý late data định kỳ hoặc drift monitoring liên tục.

## Cách kiểm tra

```bash
python -m src.pipeline train --config configs/config.yaml --no-artifacts
python -m pytest
```
