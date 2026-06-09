# Tái hiện DeepGaze IIE — Khảo sát tổ hợp Backbone

Dự án này tái hiện và khảo sát thực nghiệm mô hình dự đoán saliency thị giác **DeepGaze IIE** (Linardos et al., ICCV 2021), tập trung vào cơ chế cốt lõi tạo nên hiệu năng của mô hình: **sự kết hợp (ensemble) nhiều backbone**.

Chúng tôi tái hiện lại các mô hình tổ hợp kết hợp nhiều backbone theo đúng kiến trúc và quy trình của tác giả, đồng thời khảo sát một cách có hệ thống các mức độ kết hợp khác nhau — từ 2 backbone đến 5 backbone — với cấu hình chính là **4 backbone** mà tác giả đã chọn ra sau nhiều lần thực nghiệm (ShapeNet-C, EfficientNetB5, DenseNet201, ResNext50).

Mô hình được lấy cảm hứng từ tác giả và chỉ được tối ưu/giảm nhẹ lại cho phù hợp với giới hạn tài nguyên (RAM và GPU trên Google Colab), nhưng vẫn giữ nguyên triết lý thiết kế: backbone đóng băng, mạng readout tích chập 1×1, và kết hợp ở mức phân bố mật độ điểm nhìn qua phép logsumexp.

## Tổng quan

- **Bài toán:** Dự đoán vị trí điểm nhìn cố định (fixation) của con người trên ảnh — đầu ra là một phân bố xác suất hai chiều.
- **Mục tiêu:** Kiểm chứng độc lập nhận định của tác giả về *tính bổ trợ* giữa các backbone — liệu việc kết hợp thêm backbone có thực sự cải thiện hiệu năng hay không.
- **Cách tiếp cận:** Tái hiện pipeline đầy đủ, huấn luyện và đánh giá toàn bộ các tổ hợp backbone, so sánh và trực quan hóa kết quả.

## Kiến trúc mô hình

Mô hình kế thừa kiến trúc DeepGaze II với điều chỉnh quan trọng nhất là thay backbone đơn lẻ bằng nhiều backbone kết hợp:
Ảnh → Backbone CNN (đóng băng) → Trích xuất đặc trưng đa lớp
→ Readout (Conv 1×1 + LayerNorm + Softplus)
→ Finalizer (blur + center-bias + log-softmax)
→ Phân bố log-mật độ fixation

Cơ chế ensemble hai cấp:
- **Intra-model:** kết hợp nhiều instance readout trên cùng một backbone.
- **Inter-model:** kết hợp các backbone khác nhau.

Chỉ phần readout, sigma của blur và trọng số center-bias được huấn luyện; toàn bộ backbone giữ cố định.

## Quy trình huấn luyện

Mô hình được huấn luyện qua **hai giai đoạn (2 phase)** theo bài báo:

1. **Phase 1 — Pretrain:** tiền huấn luyện trên SALICON (~10.000 ảnh).
2. **Phase 2 — Finetune:** tinh chỉnh trên MIT1003 (~1.000 ảnh).

Do giới hạn phiên làm việc của Colab, mô hình hỗ trợ lưu checkpoint và tiếp tục huấn luyện (resume) chính xác từ điểm dừng.

## Đánh giá

Mô hình được đánh giá cả **trong miền (in-domain)** và **ngoài miền (out-of-domain)**:

| Dataset | Phân loại | Vai trò |
|---------|-----------|---------|
| SALICON | In-domain | Đánh giá sau pretrain |
| MIT1003 | In-domain | Đánh giá sau finetune |
| Toronto | Out-of-domain | Inference, không huấn luyện lại |
| PASCAL-S | Out-of-domain | Inference, không huấn luyện lại |

Thước đo sử dụng:
- **Information Gain (IG)** — thước đo chính, đo mức vượt trội so với baseline center-bias.
- **Normalized Scanpath Saliency (NSS)** — thước đo bổ trợ.

## Cấu trúc thư mục

Folder bao gồm các file code chạy các tổ hợp kết hợp backbone từ 2 đến 5 backbone: Có 6 tổ hợp 2 chập 4, 4 tổ hợp 3 chập 4, 1 tổ hợp 4 backbone chính, 1 tổ hợp 4 backbone chính kết hợp ResNet50

## Kết quả chính

- Mô hình ensemble 4 backbone đạt hiệu năng tốt và duy trì được khả năng dự đoán trên dữ liệu ngoài miền (IG dương rõ rệt trên cả Toronto và PASCAL-S).
- Thử nghiệm thêm ResNet50 (mô hình 5 backbone) **không cải thiện mà làm giảm nhẹ** hiệu năng trên MIT1003 — tái hiện đúng kết luận của bài báo rằng việc kết hợp backbone phải dựa trên *tính bổ trợ* chứ không phải số lượng.

## Tài liệu gốc

Nếu mọi người có hứng thú, đây là các nguồn gốc của mô hình: https://github.com/matthias-k/DeepGaze

## Ghi chú

Dự án này được thực hiện trong khuôn khổ học phần PBL4 với mục đích học thuật — tái hiện và kiểm chứng mô hình, không nhằm thay thế hay cải tiến công trình gốc. Mọi đóng góp về kiến trúc và phương pháp thuộc về nhóm tác giả DeepGaze IIE.
