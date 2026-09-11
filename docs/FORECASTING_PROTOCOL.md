# Quy ước đánh giá dự báo PM2.5

Tài liệu này chứa công thức và quy ước đánh giá được dùng bởi pipeline PM2.5. README chỉ mô tả contract và luồng vận hành; công thức chi tiết được giữ ở đây để tránh hai tài liệu diễn giải khác nhau.

## 1. Bài toán

Với một trạm và origin t, hệ thống dự báo:

    y_hat(t+1) = f(history up to t)

History chỉ bao gồm các quan trắc đã có sẵn tại origin. Nhãn huấn luyện là PM2.5(t+1) và không được dùng để tạo feature tại t.

## 2. Baseline bắt buộc

### Persistence

    y_hat_persistence(t+1) = y(t)

Trong code, baseline này được tạo bởi src/forecasting/baselines.py.

### Seasonal Naive 24h

Dự báo t+1 bằng quan trắc cùng giờ của ngày trước:

    y_hat_seasonal24(t+1) = y((t+1)-24h) = y(t-23h)

Vì feature origin là t, offset -23 trong build_features() là đúng với target ở t+1, không phải lỗi lệch một giờ.

## 3. Feature protocol

- Lag PM2.5 được lookup theo khóa (station_id, timestamp - lag).
- Không dùng DataFrame.shift() vì row thứ 24 có thể không cách origin đúng 24 giờ khi chuỗi bị đứt.
- Rolling mean/std dùng closed="left" để loại trừ chính quan trắc tại origin khỏi cửa sổ lịch sử.
- Delta được tính từ các lag exact-clock-time.
- Biến lịch dùng giờ địa phương TP.HCM sau khi timestamp đã được chuẩn hóa về UTC.
- Exogenous feature chỉ hợp lệ nếu nguồn có available_at <= origin. Future weather chỉ được phép khi đó là forecast source có contract riêng.

## 4. Temporal evaluation

Pipeline có ba vùng thời gian:

1. **Train:** dùng để fit model và tạo expanding-window CV.
2. **Calibration:** future window độc lập, dùng để tính residual quantile và tiêu chí chọn chiến lược.
3. **Final test:** vùng tương lai chưa được dùng để chọn model hoặc hiệu chuẩn.

Nếu một hàng feature tại t có target timestamp t+1, điều kiện biên dùng trong split là:

    target_timestamp < validation_start

Điều này ngăn hàng cuối của train có nhãn chạm vào validation hoặc calibration boundary.

## 5. Model selection

Với mỗi candidate, evaluate_candidate() chạy expanding-window folds. Model tốt nhất
được chọn theo MAE CV trung bình thấp nhất:

    best_cv_model = argmin(mean(MAE_model_fold))

Config hiện tại bật:

    model_comparison:
      candidates: [ridge, hist_gradient_boosting]

Final test không được dùng trong phép chọn này.

## 6. Metrics

### MAE và RMSE

    MAE = mean(abs(y - y_hat))
    RMSE = sqrt(mean((y - y_hat)^2))

### MASE và Skill Score

Trong repo, Persistence là scale tham chiếu:

    MASE = MAE_model / MAE_persistence
    Skill_persistence = 1 - MAE_model / MAE_persistence

MASE nhỏ hơn 1 hoặc Skill dương nghĩa là model tốt hơn Persistence trên cùng tập đánh giá.

## 7. Split Conformal Prediction

Calibration residual của model là:

    r_i = abs(y_i - y_hat_i)

Với coverage mục tiêu 1-alpha, conformal_quantile() dùng finite-sample rank:

    k = min(ceil((n + 1) * (1 - alpha)), n)

Sau khi sắp xếp residual tăng dần, q là residual ở rank k. Khoảng dự báo là:

    C(x) = [max(0, y_hat - q), y_hat + q]

Pipeline tính q toàn cục và q theo trạm khi trạm có ít nhất minimum_calibration_samples_per_station mẫu. Nếu trạm chưa đủ mẫu, bước dự báo dùng q toàn cục.

Các chỉ số interval:

- **PICP:** tỷ lệ y nằm trong [lower, upper].
- **MPIW:** trung bình upper - lower.
- Coverage trên time series chỉ là kết quả kiểm định theo window; không được diễn giải thành bảo đảm vô điều kiện cho mọi giai đoạn tương lai.

## 8. Tiêu chí chọn chiến lược

Sau khi có model tốt nhất theo CV, pipeline chỉ chọn model đó để dự báo nếu:

1. MAE trên calibration tốt hơn Persistence tối thiểu `minimum_mae_improvement`;
2. độ lệch chuẩn MAE giữa các fold không vượt `maximum_cv_mae_std`;
3. nếu có PICP calibration, khoảng cách với coverage mục tiêu không vượt
   `maximum_picp_gap`.

Nếu một điều kiện không đạt, `forecast_strategy` được đặt là `persistence`.
Final test chỉ dùng để báo cáo kết quả, không tham gia quyết định này.

Recall nhóm PM2.5 cao vẫn được ghi trong báo cáo để phân tích lỗi nghiệp vụ,
nhưng không quyết định chiến lược dự báo.

Ở bước dự báo, history ngắn, PM2.5 hiện tại bị thiếu, gap lớn hoặc thiếu biến ngoại
sinh có thể kích hoạt Persistence fallback theo chính sách trong cấu hình.
