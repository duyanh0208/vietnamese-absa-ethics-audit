# AUDIT.md: dấu vết kiểm toán cho kết quả trong `results/`

Tệp này ghi lại chính xác kết quả trong `results/` được sinh ra như thế nào:
lệnh nào, theo thứ tự nào, lúc nào, trên môi trường nào, và checksum của từng
tệp đầu ra. Mục đích là để người đánh giá đối chiếu được rằng con số trong
báo cáo đến từ lần chạy này chứ không phải từ một lần chạy cũ nào đó còn sót
lại trong thư mục.

Tệp này được ghi tay ngay sau lần chạy, từ nhật ký lệnh và từ đầu ra của
`sha256sum`. Repository không có script sinh ra nó, nên nếu `results/` được sinh
lại mà tệp này không được cập nhật thì nó sẽ lệch một cách âm thầm. Đó là một
điểm yếu đã biết của dấu vết kiểm toán này, và cách bù lại là mục 5 bên dưới:
chạy lại toàn bộ rồi đối chiếu bằng `git status --porcelain results/`.

## 1. Phiên bản mã nguồn

| Mục | Giá trị |
|---|---|
| Nhánh | `main` |
| Commit lúc chạy | `64de5fc2e1c6c25bfe547f104942a03822564107` |
| Commit rút gọn | `64de5fc` |
| Cây làm việc | CÓ THAY ĐỔI CHƯA COMMIT |

Một điểm cần nói rõ để tệp này không tự nhận nhiều hơn nó chứng minh được.
Hash `64de5fc` là commit mà lần chạy được thực hiện **từ đó**. Bản
thân `AUDIT.md` được commit sau lần chạy, nên commit chứa tệp này là commit
con của `64de5fc`, và hash của nó không thể nằm bên trong chính nó.
Muốn xác định tuyệt đối bản đã nộp thì dùng commit cuối của nhánh, hoặc tag
bản nộp và ghi tag đó vào đây.

## 2. Môi trường

Năm dòng đầu lấy trực tiếp từ `results/metrics_overall.json`, tức là do chính
`src/train_baseline.py` ghi ra lúc huấn luyện chứ không chép tay.

| Thành phần | Phiên bản |
|---|---|
| Python | 3.13.9 |
| numpy | 2.4.4 |
| pandas | 3.0.2 |
| scikit-learn | 1.8.0 |
| scipy | 1.17.1 |
| Hệ điều hành | Windows-11-10.0.26200-SP0 |
| Bộ xử lý | AMD64 |

Backend BLAS: openblas 0.3.31.188.0, 12 luồng, nhân `Haswell`.

Backend BLAS và số luồng được ghi lại vì chúng ảnh hưởng đến kết quả:
thứ tự cộng dồn dấu chấm động trong solver `lbfgs` phụ thuộc vào
chúng, và `random_state` không kiểm soát được điều đó. Xem chương 7 của
`docs/ETHICS.md`.

Siêu tham số: `seed = 42`, `C = 4.0`

## 3. Dữ liệu đầu vào

Checksum SHA-256 của ba tệp gốc trong `data/raw/`. Thư mục này bị `.gitignore`
loại theo giấy phép UIT-ViSFD, nên checksum ở đây là cách duy nhất để xác nhận
lần chạy dùng đúng phiên bản dữ liệu ghi trong `DATA.md` mục 1.1.

| Tệp | Kích thước (byte) | SHA-256 |
|---|---:|---|
| `Dev.csv` | 309.271 | `ab29bf9e7a2002013e8670c110685a1ea14eda3f87a575f01e9ccb4dd4d9da51` |
| `Test.csv` | 636.268 | `a30901f08b907f2a9df094626dbe96dcdfb01845fa7d152408aff20bc9785bd9` |
| `Train.csv` | 2.240.007 | `cd4426cfffdf574e2a46513692ddb0d343334fb4409b7850318dba479878eb7b` |

## 4. Lệnh đã chạy, theo thứ tự

Thời điểm ghi theo giờ UTC, lấy ngay trước khi lệnh bắt đầu.

| # | Thời điểm (UTC) | Thời gian chạy | Mã thoát | Lệnh |
|---:|---|---:|---:|---|
| 1 | 2026-09-07T09:04:09Z | 1s | 0 | `python src/prepare_data.py` |
| 2 | 2026-09-07T09:04:11Z | 31s | 0 | `python src/train_baseline.py` |
| 3 | 2026-09-07T09:04:42Z | 6s | 0 | `python src/fairness_check.py` |
| 4 | 2026-09-07T09:04:48Z | 3s | 0 | `python src/interpret.py` |
| 5 | 2026-09-07T09:04:51Z | 9s | 0 | `python src/sensitivity_threshold.py` |
| 6 | 2026-09-07T09:05:00Z | 17s | 0 | `python src/error_analysis.py` |
| 7 | 2026-09-07T09:05:18Z | 1s | 0 | `python src/scan_pii.py` |
| 8 | 2026-09-07T09:05:19Z | 6s | 0 | `python -m unittest discover -s tests` |

Bốn lệnh đầu là pipeline chính và thứ tự của chúng là bắt buộc: mỗi bước đọc
đầu ra của bước trước. Ba lệnh tiếp theo là các phép kiểm toán bổ sung; chúng
chỉ đọc mô hình đã huấn luyện và dữ liệu đã xử lý, không ghi đè bất cứ thứ gì
của pipeline chính, nên có chạy hay không cũng không đổi các con số ở mục 7.
Lệnh cuối chạy bộ kiểm thử và không ghi ra `results/`.

## 5. Kiểm tra tính xác định trên cùng máy

Toàn bộ chuỗi lệnh trên được chạy lại từ đầu trên cùng máy này, hai lần, cách
nhau một ngày. Cả hai lần, mọi tệp trong `results/` đều trùng khớp từng byte với
bản đã commit trước đó, kiểm bằng `git status --porcelain results/`.

Lần kiểm thứ hai được thực hiện sau khi các mục của `docs/ETHICS.md` được đánh
số lại. Việc đánh số lại chạm vào chuỗi `purpose` mà `src/scan_pii.py` và
`src/sensitivity_threshold.py` ghi vào tệp kết quả của chúng, nên đúng hai tệp đó
đổi checksum. Phần số liệu bên trong hai tệp không đổi một ký tự nào, và sáu tệp
còn lại giữ nguyên checksum cũ.

Tính xác định này chỉ có hiệu lực trên cùng máy và cùng môi trường. Khi đổi
nền tảng, kết quả lệch ở mức đã đo và ghi tại chương 7 của `docs/ETHICS.md`.

## 6. Checksum của tệp đầu ra

| Tệp | Kích thước (byte) | SHA-256 |
|---|---:|---|
| `results/error_analysis.json` | 26.784 | `39c6a8147c6133c36c7dca0f90a7b60c8097f77c65795446050f7ec03ad721a4` |
| `results/fairness_report.json` | 12.501 | `5144854a96d73bc689349a6620ad2c60c48f0186f193fe7c308ac2dfa2e63fe6` |
| `results/interpretability.json` | 3.723 | `28f8c108124b92c0a6cb079410afaa69846bf245419098ab36bc1afd2161ccec` |
| `results/metrics_overall.json` | 1.256 | `3c783c163b7cd79395e234d0570e684089a9f6a7ca73684a5a833e124b3ea332` |
| `results/model_baseline.joblib` * | 7.471.608 | `038f00b7a0180b5e9056f5647e8192324703a09e26474f6f50e4ad46e277f5fa` |
| `results/pii_scan.json` | 3.017 | `1be793e873b578351932c1132b156d587401d64c8dd4d83e6ed59a26a7d9b906` |
| `results/predictions_test.csv` | 1.784.563 | `09be7e8d80dad3cd3cefe0291c4756edc06d5e230ee08b970decccfeb1a81d94` |
| `results/sensitivity_threshold.json` | 25.886 | `0fbd7e2f723ca36c50d3423a80b6d4100e5fd22921a3af10e321c27d239a0d5c` |

`*` = bị `.gitignore` loại, không nằm trong repository. Tệp mô hình được
sinh lại bằng `python src/train_baseline.py`; checksum ghi ở đây để đối
chiếu khi sinh lại.

## 7. Con số chính của lần chạy này

| Chỉ số | Giá trị |
|---|---|
| Macro-F1 (test) | 0.713079 |
| Accuracy (test) | 0.785332 |
| F1 lớp Negative | 0.804242 |
| F1 lớp Neutral | 0.484615 |
| F1 lớp Positive | 0.850380 |
| EO gap theo aspect | 0.474587 (CI 95% [0.365; 0.588]) |
| EO gap theo số sao | 0.311179 (CI 95% [0.269; 0.352]) |
| EO gap theo độ dài | 0.094280 (CI 95% [0.059; 0.138]) |
| EO gap theo mức chuẩn chính tả | 0.053704 (CI 95% [0.025; 0.099]) |

Các giá trị này là nguồn duy nhất được coi là đúng cho lần chạy hiện tại.
Nếu một con số trong `README.md` hoặc `docs/ETHICS.md` không khớp với bảng
trên, thì tài liệu đang mô tả một lần chạy khác và phải được đối chiếu lại
trước khi nộp.
