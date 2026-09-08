# ABSA tiếng Việt với kiểm toán công bằng và diễn giải

`https://github.com/duyanh0208/vietnamese-absa-ethics-audit`

Dự án cho môn AI Ethics (MCSE, Vietnam Japan University). Bài toán: Aspect
Category Sentiment Analysis (ACSA) trên phản hồi người dùng tiếng Việt, kèm một
lớp kiểm toán công bằng và diễn giải được xây dựng như thành phần bắt buộc của
pipeline chứ không phải phần bổ sung sau khi có kết quả.

Bộ dữ liệu: UIT-ViSFD (11.122 bình luận, 10 aspect × 3 polarity). Chi tiết nguồn,
giấy phép và tiền xử lý xem `DATA.md`.

Phân tích đạo đức đầy đủ, chỉ số công bằng được đề xuất, checklist tự đánh giá và
phương án bị loại vì lý do đạo đức nằm trong `docs/ETHICS.md`.

## Về bản báo cáo

Repository này chứa phần mã nguồn và kết quả nghiên cứu. Bản báo cáo phân tích
đạo đức là bài nộp của môn học và được nộp trực tiếp cho giảng viên, nên nó
không nằm ở đây. Các tham chiếu tới `docs/ETHICS.md` trong tài liệu dưới đây
trỏ tới bản báo cáo đó.

Phần nằm ở đây là toàn bộ những gì cần để tự kiểm chứng: mã nguồn, bộ kiểm thử,
kết quả đã sinh trong `results/`, tài liệu hóa dữ liệu ở `DATA.md`, và dấu vết
kiểm toán ở `results/AUDIT.md`. Mọi con số trong báo cáo đều sinh ra từ đúng
những tệp này, và checksum trong `results/AUDIT.md` là thứ neo chúng lại.

Các commit ở đây chia theo trật tự phụ thuộc của pipeline, không theo trình tự
thời gian: repository được tạo sau khi công việc đã xong. Chúng nói phần nào
đứng trên phần nào, không nói công việc đã diễn ra theo thứ tự nào.

## Tái lập kết quả

Yêu cầu: Python 3.10+, không cần GPU, không cần tải mô hình từ internet.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

bash scripts/download_data.sh        # tải UIT-ViSFD, in checksum SHA-256
python src/prepare_data.py           # data/raw -> data/processed
python src/train_baseline.py         # huấn luyện + sinh predictions_test.csv
python src/fairness_check.py         # kiểm toán công bằng
python src/interpret.py              # diễn giải mức toàn cục
```

Toàn bộ chạy dưới 5 phút trên CPU. Mọi bước dùng `random_state=42`; chạy lại
trên cùng một máy cho cùng một kết quả. Giới hạn của tính tái lập khi đổi máy
được nêu ở chương 7 của `docs/ETHICS.md`.

### Ghi chú cho người dùng Windows PowerShell

Hai chỗ trong hướng dẫn trên cần đổi khi chạy bằng PowerShell.

**Kích hoạt môi trường ảo** dùng script của Windows thay cho `source`:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Khi gọi `bash`, đường dẫn phải dùng gạch chéo xuôi.** PowerShell chấp nhận
gạch chéo ngược, nhưng `bash` thì không: dấu `\` là ký tự thoát trong shell của
nó, nên `scripts\download_data.sh` bị đọc thành `scriptsdownload_data.sh` và
lệnh báo không tìm thấy tệp. Viết đúng:

```powershell
bash scripts/download_data.sh
```

Các lệnh `python src/...` chạy được với cả hai kiểu gạch chéo vì Python tự xử lý,
nhưng dùng gạch chéo xuôi cho toàn bộ tài liệu là nhất quán hơn.

**`unzip` phải có trong PATH.** Bản Git for Windows mặc định không kèm `unzip`,
nên `scripts/download_data.sh` dừng ở dòng giải nén với thông báo
`unzip: command not found`. Kiểm tra bằng `bash -lc "command -v unzip"`. Nếu
thiếu, cách nhanh nhất là tải ba tệp `Train.csv`, `Dev.csv`, `Test.csv` từ
`UIT-ViSFD.zip` trong repository gốc rồi đặt thủ công vào `data/raw/`, sau đó
đối chiếu checksum trong `DATA.md` mục 1.1 để xác nhận đúng phiên bản dữ liệu.

Diễn giải một câu cụ thể:

```bash
python src/interpret.py --text "pin tụt nhanh quá, nhưng chụp ảnh thì rất đẹp" \
                        --aspect BATTERY
```

## Kết quả

Baseline TF-IDF + Logistic Regression trên tập test:

| Chỉ số | Giá trị |
|---|---|
| Macro-F1 | 0,713 |
| Accuracy | 0,785 |
| F1 lớp Positive | 0,850 |
| F1 lớp Negative | 0,804 |
| F1 lớp Neutral | 0,485 |

Kiểm toán công bằng, đo bằng equal opportunity gap trên lớp Negative:

| Chiều phân nhóm | EO gap | CI 95% | Nhóm yếu nhất |
|---|---|---|---|
| Aspect | 0,475 | [0,365; 0,588] | PRICE (TPR 0,456; CI [0,346; 0,562]) |
| Số sao đánh giá | 0,311 | [0,269; 0,352] | 4-5 sao (TPR 0,608; CI [0,570; 0,647]) |
| Độ dài văn bản (tứ phân vị) | 0,094 | [0,059; 0,138] | Q4, 49-205 từ (TPR 0,774; CI [0,741; 0,806]) |
| Mức chuẩn chính tả (tứ phân vị tỷ lệ từ không dấu) | 0,054 | [0,025; 0,099] | Q3, 22-28% (TPR 0,798; CI [0,761; 0,832]) |

Khoảng tin cậy là percentile 95% từ bootstrap 2.000 lần, resample có hoàn lại ở
mức mẫu, seed 42 (`src/fairness_check.py`, hàm `bootstrap_gap_ci`). Khoảng của
chiều aspect rộng vì nhóm yếu nhất chỉ có 79 mẫu Negative: ở cỡ mẫu đó, một mẫu
đổi nhãn đã làm TPR dịch 1/79 = 0,013, nên chữ số thứ ba của gap không mang
thông tin.

Con số quan trọng nhất không phải macro-F1 mà là khoảng cách 0,475 giữa các
aspect. Diễn giải và hệ quả nằm ở mục 2.5 của `docs/ETHICS.md`.

## Thiết kế

**Vì sao baseline tuyến tính.** Mô hình chính của luận văn là fine-tune LLM sinh
tuple với LoRA/DoRA, cần GPU. Repository này cố tình dùng một baseline chạy được
trên máy bất kỳ, vì tiêu chí reproducibility của môn học yêu cầu người đánh giá
tự chạy lại được. Mô hình tuyến tính còn cho phép phân rã đóng góp đặc trưng
chính xác thay vì xấp xỉ như LIME hay Kernel SHAP.

**Vì sao không tách từ.** Bộ dữ liệu chứa nhiều teencode, viết tắt và lỗi chính
tả. Công cụ tách từ tiếng Việt được huấn luyện trên văn bản chuẩn nên tạo lỗi
phân đoạn trên loại văn bản này. Char n-gram (2-5) trong `char_wb` xử lý được
biến thể chính tả mà không cần giả định về chuẩn ngôn ngữ của người viết.

Lựa chọn này được kỳ vọng có ý nghĩa công bằng, và kỳ vọng đó đã được đem đi đo
thay vì được nêu như kết luận. Chiều phân nhóm `writing_bucket` chia tập test
theo tứ phân vị tỷ lệ từ không dấu, và cho EO gap 0,054, nhỏ nhất trong bốn
chiều đã đo. Nhưng khoảng tin cậy [0,025; 0,099] không chứa 0, nên phát biểu
đúng là **chênh lệch nhỏ chứ không phải không có**: nhóm viết chuẩn nhất đạt TPR
0,852 còn nhóm thấp nhất đạt 0,798. Chi tiết ở mục 2.5 của `docs/ETHICS.md`.

**Vì sao `n_star` không phải đặc trưng.** Xem `DATA.md` mục 2.

## Cấu trúc

```
├── DATA.md                    nguồn dữ liệu, giấy phép, tiền xử lý
├── scripts/download_data.sh   tải dữ liệu + checksum
├── src/prepare_data.py        phân tích nhãn, sinh cặp (bình luận, aspect)
├── src/train_baseline.py      huấn luyện, sinh dự đoán
├── src/fairness_check.py      kiểm toán công bằng (equal opportunity + bootstrap CI)
├── src/interpret.py           diễn giải toàn cục và cục bộ
├── src/sensitivity_threshold.py  kiểm chứng phương án bị loại ở mục 6.2
├── src/error_analysis.py      phân loại nguyên nhân dự đoán sai, có nhóm đối chứng
├── src/scan_pii.py            quét thông tin định danh trong nội dung bình luận
├── tests/                     kiểm thử cho fairness_check và sensitivity_threshold
├── results/AUDIT.md           dấu vết kiểm toán: lệnh, thời điểm, checksum
└── results/                   metrics, dự đoán, báo cáo công bằng
```

Ba script kiểm toán dưới đây không nằm trong pipeline chính và không sửa mô hình.
Chúng chỉ đọc kết quả đã có, nên chạy hay không chạy đều không ảnh hưởng tới các
con số ở trên.

```bash
python src/sensitivity_threshold.py   # -> results/sensitivity_threshold.json
python src/error_analysis.py          # -> results/error_analysis.json
python src/scan_pii.py                # -> results/pii_scan.json
```

## Kiểm thử

```bash
python -m unittest discover -s tests -v
```

34 kiểm thử, dùng `unittest` của thư viện chuẩn nên không thêm phụ thuộc nào vào
`requirements.txt`. Chúng tập trung vào các bất biến có thể hỏng âm thầm mà
không làm script báo lỗi: ngưỡng `MIN_GROUP_SIZE` là biên chặt, bootstrap cùng
seed cho cùng kết quả, khoảng tin cậy hẹp lại khi cỡ mẫu tăng, và
`apply_threshold` không bao giờ đụng vào dòng ngoài nhóm mục tiêu. Kiểm thử tích
hợp cuối cùng xác nhận `build_test_matrix` tái lập đúng `predictions_test.csv`,
tức là nó chỉ transform chứ không vô tình khớp lại vectorizer; kiểm thử này tự
bỏ qua nếu chưa chạy pipeline.

Bộ kiểm thử đã được kiểm chứng bằng cách cố tình gây lỗi: năm đột biến trong
`src/` (đổi `>=` thành `>` ở ngưỡng nhóm nhỏ, cho `apply_threshold` ghi đè cả
mảng, tính precision trên toàn bộ mẫu, bỏ seed của bootstrap, đảo logic đếm từ
không dấu) đều bị bắt.

`sensitivity_threshold.py` đo cái giá của phương án đã bị loại ở mục 6.2 của
`docs/ETHICS.md`. `error_analysis.py` lấy mẫu 50 dự đoán sai và 50 dự đoán đúng
làm đối chứng, rồi tách đóng góp vào logit theo loại đặc trưng. `scan_pii.py`
quét PII và **không bao giờ** ghi nội dung bình luận thô vào tệp kết quả; muốn
xem bản gốc để thẩm định thì dùng `--show-raw`, chỉ in ra màn hình.

## Giới hạn đã biết

Baseline không dùng ngữ cảnh, nên các bình luận có tương phản trong câu
("pin kém nhưng camera đẹp") được xử lý bằng n-gram thay vì bằng biểu diễn ngữ
nghĩa. Đây là một phần lý do lớp Neutral có F1 thấp.

Chiều độ dài văn bản được chia theo tứ phân vị của chính tập đánh giá, không
theo ngưỡng cố định, vì ngưỡng cố định tạo ra nhóm quá nhỏ trên bộ dữ liệu này.

Các biến phân nhóm dùng để kiểm toán (aspect, số sao, độ dài, mức chuẩn chính
tả) là biến proxy sẵn
có trong dữ liệu, không phải thuộc tính nhân khẩu học. Kết luận về công bằng vì
vậy giới hạn ở phạm vi các proxy này. Lý do không dùng thuộc tính nhân khẩu học
và hệ quả của giới hạn đó được trình bày ở mục 2.3 của `docs/ETHICS.md`.

Mọi con số trong báo cáo đều đến từ baseline chạy được trong repository này.
Không có kết quả nào được lấy từ mô hình chưa chạy hoặc từ tài liệu khác.
