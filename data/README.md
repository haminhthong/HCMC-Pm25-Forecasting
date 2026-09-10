# Dữ liệu mẫu

Thư mục này chứa dữ liệu đầu vào nhỏ dùng để kiểm thử toàn bộ pipeline dự báo
PM2.5. Dữ liệu mẫu không đại diện cho chất lượng không khí toàn thành phố và
không được dùng để kết luận hiệu năng thực tế.

## Nguồn hiện tại

Pipeline V1 chỉ đọc một file CSV được khai báo trong `configs/config.yaml`:

```text
data/sample/air_quality_sample.csv
```

Các API trạm và API thời tiết chưa thuộc pipeline canonical. Khi bổ sung nguồn
mới, dữ liệu phải được chuyển về đúng schema trước khi đưa vào bước kiểm tra
chất lượng.

## Schema bắt buộc

| Cột | Vai trò | Quy ước |
|---|---|---|
| `timestamp` | Thời điểm quan trắc | ISO-8601; được chuẩn hóa về UTC nội bộ |
| `station_id` | Định danh trạm | Chuỗi, không rỗng |
| `PM2.5` | Nồng độ PM2.5 | Mục tiêu tại `t+1`, đơn vị `µg/m³`, không âm |

Các cột `TSP`, `NO2`, `SO2`, `CO`, `O3`, `temperature`, `humidity` là biến
ngoại sinh tùy chọn. Chỉ những cột được khai báo trong
`features.exogenous_columns` mới được dùng làm feature.

## Quy ước thời gian và dữ liệu thiếu

- Timestamp được parse theo `data.source_timezone`, sau đó chuyển sang UTC.
- Mỗi trạm được regularize về lưới một giờ. Giờ không có quan trắc được chèn
  rõ ràng với giá trị thiếu.
- Lag được lookup theo `(station_id, timestamp - n giờ)`, không dùng `shift()`
  theo vị trí dòng.
- Rolling feature chỉ dùng quan trắc trước forecast origin; không dùng giá trị
  của chính origin hoặc tương lai.
- Bản ghi trùng timestamp trong cùng trạm bị từ chối.
- Dữ liệu không đúng thứ tự được sắp xếp theo trạm và thời gian trước khi xử lý.
- Thiếu biến ngoại sinh, gap lớn hoặc lịch sử không đủ sẽ làm dự báo runtime
  chuyển sang Persistence; lịch sử ngắn hơn mức tối thiểu có thể bị từ chối.

## Hợp đồng availability

Trong pipeline hiện tại, `feature_availability` trong cấu hình mô tả độ trễ
phát hành của từng biến. Một feature tại thời điểm `t` chỉ hợp lệ khi đã
available tại forecast origin. Giá trị tương lai như `weather(t+1)` không được
dùng, trừ khi có nguồn dự báo thời tiết riêng được tích hợp và kiểm thử.

## Phạm vi dữ liệu

Đây là sample/smoke data phục vụ test, demo API và dashboard. Các chỉ số trong
`artifacts/evaluation.json` chỉ có ý nghĩa kiểm tra đường đi của hệ thống;
không phải benchmark HCMC citywide.
