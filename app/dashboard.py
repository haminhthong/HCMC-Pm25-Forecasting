"""Dashboard Streamlit minh họa kết quả dự báo PM2.5 giờ tiếp theo."""

import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from src.data import load_air_quality, load_config

st.set_page_config(page_title="Dự báo PM2.5 TP.HCM", layout="wide")
st.title("Dự báo PM2.5 giờ tiếp theo (Next-Hour PM2.5 Forecasting)")
st.caption(
    "Dự án portfolio thử nghiệm. Các mức Thấp/Trung bình/Cao là nhóm phân tích nội bộ, "
    "không phải chỉ số AQI chính thức hoặc khuyến nghị y tế."
)

config = load_config("configs/config.yaml")
api_url = os.getenv("PM25_API_URL", "http://localhost:8000")
data = load_air_quality(config)
station_column = config["data"]["station_column"]
timestamp_column = config["data"]["timestamp_column"]
target = config["data"]["target_column"]

station = st.selectbox("Trạm quan trắc", sorted(data[station_column].unique()))
history = data[data[station_column] == station].sort_values(timestamp_column).tail(25)

if len(history) < 25:
    st.error("Trạm được chọn không đủ 25 giờ quan trắc liên tục.")
else:
    try:
        payload_history = history.copy()
        payload_history[timestamp_column] = payload_history[timestamp_column].astype(str)
        payload_history = payload_history.astype(object).where(pd.notna(payload_history), None)
        response = requests.post(
            f"{api_url}/predict",
            json={"observations": payload_history.to_dict("records")},
            timeout=10,
        )

        if response.status_code == 200:
            result = response.json()
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("PM2.5 Hiện tại (t)", f"{result['current_pm25']:.1f} µg/m³")
            c2.metric("Dự báo Giờ tới (t+1)", f"{result['predicted_pm25']:.1f} µg/m³")
            c3.metric("Mức phân tích", result["level"])

            interval = result.get("interval", {})
            interval_str = (
                f"[{interval.get('lower', 0):.1f} - {interval.get('upper', 0):.1f}] µg/m³"
            )
            c4.metric("Khoảng Conformal", interval_str)
            width = interval.get(
                "width",
                round(interval.get("upper", 0) - interval.get("lower", 0), 2),
            )
            c5.metric("Độ rộng khoảng (MPIW)", f"±{width / 2:.1f} µg/m³")

            strategy = result.get("forecast_strategy", "persistence")
            best_model = result.get("best_cv_model", "unknown")
            if strategy == "persistence":
                badge = "🟠 **Chiến lược:** `persistence` (history không đủ tốt cho model)"
            else:
                badge = f"🟢 **Chiến lược:** `{strategy}` (best CV model: `{best_model}`)"

            st.markdown(
                f"{badge} · **Phạm vi dữ liệu:** `{result.get('dataset_scope', 'sample')}` · "
                f"**Origin (t):** `{result.get('forecast_origin')}` · "
                f"**Target (t+1):** `{result.get('forecast_for')}`"
            )
            # Vẽ biểu đồ chuỗi thời gian lịch sử và mốc dự báo
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=history[timestamp_column],
                    y=history[target],
                    mode="lines+markers",
                    name="Quan trắc lịch sử",
                    line={"color": "#1f77b4"},
                )
            )

            target_time = pd.to_datetime(result.get("forecast_for"))
            pred_val = result["predicted_pm25"]
            lower_val = interval.get("lower", pred_val)
            upper_val = interval.get("upper", pred_val)

            fig.add_trace(
                go.Scatter(
                    x=[target_time],
                    y=[pred_val],
                    mode="markers",
                    name="Dự báo (t+1)",
                    marker={"color": "#d62728", "size": 12, "symbol": "diamond"},
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=[upper_val - pred_val],
                        arrayminus=[pred_val - lower_val],
                        color="#d62728",
                        width=6,
                    ),
                )
            )
            fig.update_layout(
                title=f"Chuỗi quan trắc và điểm dự báo tại {station}",
                xaxis_title="Thời gian (Timestamp)",
                yaxis_title="Nồng độ PM2.5 (µg/m³)",
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)

        elif response.status_code == 422:
            st.error(f"Lỗi Schema Validation (422): {response.json().get('detail')}")
        elif response.status_code == 400:
            st.warning(f"Dữ liệu đầu vào không hợp lệ (400): {response.json().get('message')}")
        elif response.status_code == 503:
            st.warning("Mô hình hoặc dịch vụ API chưa sẵn sàng (503).")
        else:
            st.error(f"Lỗi API HTTP {response.status_code}")

    except requests.RequestException:
        st.warning(
            "Không thể kết nối đến API server. Hãy đảm bảo API đang chạy tại http://localhost:8000"
        )
