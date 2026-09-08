# 📋 Báo Cáo Đánh Giá Hệ Thống Dự Báo PM2.5 Giờ Tiếp Theo

## 1. Kết Quả Expanding-Window Backtest (Tập Train)

| Mô hình ứng viên | MAE CV trung bình | Độ lệch chuẩn (Std) | RMSE CV trung bình |
|---|---:|---:|---:|
| `ridge` | 0.381 | 0.110 | 0.397 |
| `hist_gradient_boosting` | 4.110 | 0.552 | 4.739 |

## 2. Đánh Giá Quality Gate & Quyết Định Serving Champion

- Quality gate: **đạt**
- Cải thiện MAE vs Persistence: **67.9%** (Yêu cầu: $\ge 5\%$)
- Recall nhóm PM2.5 cao: **100.0%** (diagnostic, không phải điều kiện release)
- Độ lệch chuẩn Rolling MAE: **0.110** (Yêu cầu: $\le 1.0$)
- **Chính sách phục vụ suy luận (Serving Champion):** `ridge`

## 3. Đánh Giá Tập Test Cuối (Freeze-Policy Final Test)

| Mô hình / Chiến lược | MAE Test | RMSE Test | MASE | Skill vs Pers | Macro-F1 | Recall PM2.5 Cao |
|---|---:|---:|---:|---:|---:|---:|
| **Ứng viên ML (ridge)** | 0.316 | 0.317 | 1.264 | -26.4% | 0.667 | 100.0% |
| **Persistence Baseline (t+1 = t)** | 0.250 | 0.292 | 1.000 | 0.0% | 0.667 | 100.0% |
| **Seasonal Naive 24h (t+1 = t-23h)** | 5.948 | 5.952 | 23.790 | -2279.0% | 0.222 | 0.0% |
| 🏆 **Actual Serving Champion (ridge)** | 0.316 | 0.317 | 1.264 | -26.4% | 0.667 | 100.0% |

## 4. Kiểm Định Khoảng Tin Cậy Conformal (90% Target Coverage)

- **Độ phủ thực tế trên tập Test (PICP):** 75.0%
- **Độ rộng khoảng trung bình (MPIW):** N/A µg/m³ (±0.000 µg/m³)
- **Độ rộng khoảng trung vị:** N/A µg/m³

## 5. Đánh Giá Phân Rã Theo Trạm Quan Trắc

| Trạm quan trắc | MAE Test | RMSE Test | PICP Conformal | Độ rộng khoảng (MPIW) |
|---|---:|---:|---:|---:|
| `Trạm A` | 0.318 | 0.319 | 75.0% | ±0.328 µg/m³ |
| `Trạm B` | 0.313 | 0.314 | 75.0% | ±0.328 µg/m³ |

> ⚠️ **Lưu ý:** Báo cáo này áp dụng quy trình đánh giá chuẩn mực chống rò rỉ dữ liệu (Nested Temporal Evaluation). Khi áp dụng dữ liệu thực tế tại TP.HCM, cần kiểm tra nguồn và giấy phép cung cấp dữ liệu.