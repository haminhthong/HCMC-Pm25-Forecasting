# Data Card: Ho Chi Minh City PM2.5 Observation Dataset

## 1. Trạng thái Dữ liệu (Provenance & Readiness)

> [!WARNING]
> **SMOKE TEST ONLY — NOT A PRODUCTION BENCHMARK**
>
> Tập dữ liệu commit trong repository (`data/sample/air_quality_sample.csv`) chỉ là smoke dataset tổng hợp. Số dòng và metric phải đọc từ artifact/manifest, không hard-code trong tài liệu.
> **Tuyệt đối không sử dụng kết quả đánh giá trên file này để tuyên bố chất lượng dự báo thực tế** (như QWK, Recall, hay MASE). Toàn bộ artifact sinh ra từ tập sample được gắn cờ rõ ràng:
> `production_readiness: "smoke_test_only"`.

## 2. Temporal Contract & Chống Data Leakage

- **Forecast Origin ($t$):** Thời điểm hiện tại tại lúc thực hiện dự báo.
- **Dự báo mục tiêu ($t+1$):** Nồng độ PM2.5 trung bình tại giờ tiếp theo.
- **Tính hợp lệ của PM2.5($t$):** Quan sát PM2.5 tại mốc $t$ đã hoàn tất đo đạc tại thời điểm phát lệnh dự báo, do đó là **đặc trưng đầu vào hoàn toàn hợp lệ**, không phải rò rỉ dữ liệu.
- **Closed='left' Rolling History:** Các đặc trưng rolling (mean, std) chỉ tính trên lịch sử quan trắc nghiêm ngặt TRƯỚC thời điểm $t$.
- **Exact Clock-Time Lag:** Tra cứu theo mốc thời gian thực (`timestamp - lag`), không dịch chuyển số dòng (row-position shift). Nếu xuất hiện khoảng trống dữ liệu, lag nhận giá trị `NaN`.

## 3. Canonical Data Contract (`AirQualityDataset`)

Dữ liệu quan trắc từ mọi nguồn (CSV, OpenAQ API, Weather API) đều được chuẩn hóa về schema chung:

| Cột | Ý nghĩa | Miền giá trị vật lý | Bắt buộc |
|---|---|---|---|
| `timestamp` | Thời gian quan trắc (ISO-8601 / UTC+7) | Datetime hợp lệ | Có |
| `station_id` | Định danh trạm quan trắc | String | Có |
| `PM2.5` | Nồng độ bụi mịn PM2.5 ($\mu g/m^3$) | $0 - 1000$ | Có |
| `latitude` / `longitude` | Tọa độ địa lý của trạm | $[-90, 90]$ / $[-180, 180]$ | Tùy chọn |
| `TSP`, `PM10`, `NO2`, `SO2`, `CO`, `O3` | Nồng độ các chất ô nhiễm khác | Miền vật lý tương ứng | Tùy chọn |
| `temperature`, `humidity` | Nhiệt độ ($^\circ C$), Độ ẩm tương đối (%) | $[-20, 60]$, $[0, 100]$ | Tùy chọn |
| `wind_speed`, `wind_direction`, `rainfall` | Khí tượng bề mặt | $[0, 100]$, $[0, 360]$, $[0, 500]$ | Tùy chọn |

## 4. Quy ước Missing và Regularization

- **Hourly Regularization:** Chuỗi quan trắc của mỗi trạm được chuẩn hóa về lưới 1 giờ (`freq="h"`). Giờ bị thiếu được chèn dòng `NaN` tường minh.
- **Imputation:** Sklearn `SimpleImputer` xử lý các giá trị `NaN` đồng nhất giữa quy trình huấn luyện và phục vụ suy luận (`Predictor`).
- **Data Availability Contract:** Các biến khí tượng/ô nhiễm có độ trễ cập nhật được định nghĩa độ trễ trong cấu hình (`feature_availability`) để mô phỏng chính xác độ trễ thực tế tại origin $t$.
