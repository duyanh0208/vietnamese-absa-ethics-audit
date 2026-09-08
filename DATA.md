# DATA.md: nguồn dữ liệu, giấy phép và tiền xử lý

Tài liệu này đáp ứng tiêu chí *Code Documentation & Interpretability* của rubric:
mọi nguồn dữ liệu, giấy phép sử dụng và bước tiền xử lý đều được ghi lại đầy đủ
để người đánh giá tái lập được kết quả.

## 1. Nguồn dữ liệu

### 1.1. Thông tin bộ dữ liệu

| Mục | Nội dung |
|---|---|
| Tên | UIT-ViSFD (Vietnamese Smartphone Feedback Dataset) |
| Tác giả | Luong Luc Phan, Phuc Huynh Pham, Kim Thi-Thanh Nguyen, Sieu Khai Huynh, Tham Thi Nguyen, Luan Thanh Nguyen, Tin Van Huynh, Kiet Van Nguyen |
| Đơn vị | University of Information Technology, VNU-HCM |
| Công bố | KSEM 2021, Part II, Springer LNCS vol. 12816, tr. 647-658, doi:10.1007/978-3-030-82147-0_53 |
| Repository | https://github.com/LuongPhan/UIT-ViSFD |
| Quy mô | 11.122 bình luận (Train 7.786 / Dev 1.112 / Test 2.224) |
| Schema nhãn | 10 aspect × 3 polarity (Positive, Neutral, Negative) |
| Nguồn gốc dữ liệu thô | Bình luận của người dùng về điện thoại di động, thu thập từ một sàn thương mại điện tử lớn tại Việt Nam |

Checksum SHA-256 của ba tệp gốc được in ra bởi `scripts/download_data.sh`, cho
phép xác nhận rằng phiên bản dữ liệu dùng trong thí nghiệm trùng khớp với bản
được tải về:

```
cd4426cf...78eb7b  Train.csv
ab29bf9e...d9da51  Dev.csv
a30901f0...85bd9   Test.csv
```

### 1.2. Giấy phép và điều kiện sử dụng

Repository gốc không đính kèm tệp LICENSE theo chuẩn SPDX. Điều kiện sử dụng
được nêu trực tiếp trong README của tác giả và bao gồm ba ràng buộc:

1. Bộ dữ liệu được cung cấp miễn phí **cho mục đích nghiên cứu**.
2. Mọi công trình sử dụng dữ liệu phải trích dẫn bài báo KSEM 2021.
3. Sử dụng cho **mục đích thương mại** phải liên hệ nhóm tác giả để xin phép.

Dự án này là bài tập học thuật không thương mại, nằm trong phạm vi (1) và tuân
thủ (2) qua mục Tài liệu tham khảo của báo cáo. Ràng buộc (3) được ghi nhận ở
đây vì nó có hệ quả trực tiếp: nếu kết quả của dự án được đưa vào một sản phẩm
social listening bán ra thị trường, giấy phép hiện tại không cho phép, và đây
là một trong những rủi ro pháp lý được phân tích trong báo cáo ethics.

Vì repository không nêu giấy phép chuẩn hóa, phạm vi các quyền không được liệt
kê (phân phối lại, tạo tác phẩm phái sinh, huấn luyện mô hình rồi phát hành
trọng số) là **không xác định**. Dự án này xử lý trường hợp đó theo hướng thận
trọng: không phân phối lại dữ liệu thô trong repository (xem `.gitignore`), chỉ
cung cấp script tải về từ nguồn gốc.

### 1.3. Quy trình gán nhãn và độ đồng thuận giữa người gán nhãn

Thông tin dưới đây trích từ mục 3 của bài báo gốc, bản arXiv 2105.15079, cùng
nội dung với bản KSEM 2021 nhưng truy cập được không cần đăng nhập.

| Mục | Nội dung |
|---|---|
| Số người gán nhãn | 5 |
| Chỉ số đồng thuận | Cohen's Kappa |
| Quy trình huấn luyện | 6 vòng, mỗi vòng 200 bình luận |
| Ngưỡng chấp nhận | Kappa của **mọi nhãn** phải vượt 80% trước khi gán nhãn độc lập |
| Xử lý bất đồng | Thảo luận và bỏ phiếu, sau đó bổ sung vào hướng dẫn |

Có một điểm phải đọc kỹ và không nên trích dẫn tắt. Con số "trên 80%" là mức
đồng thuận đạt được trong **giai đoạn huấn luyện người gán nhãn**, trên các vòng
200 bình luận, và nó là *điều kiện để được bắt đầu* gán nhãn độc lập. Bài báo
không báo cáo một chỉ số Kappa đo trên chính 11.122 bình luận đã phát hành. Biểu
đồ Hình 1 của bài báo vẽ Kappa qua sáu vòng huấn luyện, nhưng các giá trị nằm
trong hình chứ không có trong văn bản.

Hệ quả cho luận văn: có thể phát biểu rằng bộ dữ liệu có quy trình kiểm soát
chất lượng nhãn tường minh với ngưỡng định lượng, nhưng **không** thể phát biểu
rằng độ đồng thuận trên bộ dữ liệu phát hành là 80%. Hai câu đó khác nhau, và
câu thứ hai là một suy diễn vượt quá điều bài báo nói. Đây là giới hạn của bộ dữ
liệu, được ghi lại thay vì bỏ qua.

### 1.4. Vấn đề đồng thuận của người dùng

Dữ liệu là bình luận công khai của người dùng thật trên sàn thương mại điện tử.
Bài báo gốc không mô tả cơ chế xin đồng thuận từ người viết bình luận. Trạng
thái "công khai" không đồng nghĩa với "đã đồng thuận cho mục đích nghiên cứu và
huấn luyện mô hình". Hệ quả cụ thể được phân tích tại chương 2 của báo cáo ethics.

## 2. Dữ liệu không sử dụng và lý do

Repository gốc cung cấp bốn trường: `comment`, `n_star`, `date_time`, `label`.

| Trường | Sử dụng | Lý do |
|---|---|---|
| `comment` | Có | Đầu vào của mô hình |
| `label` | Có | Nhãn giám sát |
| `n_star` | Chỉ dùng để phân nhóm khi kiểm toán công bằng | Xem ghi chú bên dưới |
| `date_time` | Không | Không cần cho tác vụ; giữ lại làm tăng khả năng định danh lại |

Ghi chú về `n_star`: trường này **không** được đưa vào tập đặc trưng huấn luyện,
dù nó có tương quan mạnh với polarity. Đưa vào sẽ tạo một shortcut khiến mô hình
học cách đọc số sao thay vì đọc nội dung, và làm hỏng chính phép kiểm toán công
bằng dùng `n_star` làm biến phân nhóm. Trường này chỉ xuất hiện ở khâu đánh giá.

Trường `date_time` bị loại bỏ khỏi dữ liệu đã xử lý. Kết hợp dấu thời gian với
nội dung bình luận làm tăng đáng kể khả năng đối chiếu ngược một bản ghi với một
tài khoản cụ thể trên sàn thương mại điện tử gốc.

## 3. Các bước tiền xử lý

Toàn bộ nằm trong `src/prepare_data.py`, chạy bằng một lệnh và cho kết quả xác
định (deterministic), không có bước ngẫu nhiên.

1. **Phân tích chuỗi nhãn.** Nhãn gốc có dạng `{BATTERY#Negative};{GENERAL#Positive};{OTHERS};`.
   Biểu thức chính quy `\{([^#{}]+)(?:#([^{}]+))?\}` tách từng cặp aspect–polarity.

2. **Loại nhãn ngoài schema.** Thẻ `{OTHERS}` biểu thị bình luận không thuộc
   aspect nào trong 10 aspect chuẩn; các cặp này bị loại. Mọi aspect không nằm
   trong danh sách 10 aspect gốc cũng bị loại.

3. **Mở rộng thành cặp.** Mỗi bình luận sinh ra một dòng cho mỗi aspect được gán
   nhãn. Một bình luận đề cập ba aspect trở thành ba mẫu huấn luyện.

4. **Loại bình luận rỗng** sau khi chuẩn hóa khoảng trắng.

5. **Sinh metadata phục vụ kiểm toán**: `n_chars`, `n_words` để phân nhóm theo
   độ dài văn bản.

Không áp dụng: tách từ (word segmentation), chuẩn hóa teencode, loại stopword,
lowercase thủ công. Lý do được nêu trong `README.md` mục Thiết kế.

### Kết quả sau tiền xử lý

| Tập | Bình luận | Cặp (bình luận, aspect) |
|---|---|---|
| Train | 7.786 | 23.872 |
| Dev | 1.112 | 3.316 |
| Test | 2.224 | 6.722 |

### Phân bố nhãn trên tập huấn luyện

| Polarity | Số cặp | Tỷ lệ |
|---|---:|---:|
| Positive | 13.505 | 56,6% |
| Negative | 7.464 | 31,3% |
| Neutral | 2.903 | 12,2% |

| Aspect | Số cặp |
|---|---:|
| GENERAL | 4.866 |
| PERFORMANCE | 4.140 |
| BATTERY | 3.604 |
| FEATURES | 2.642 |
| CAMERA | 2.146 |
| PRICE | 2.061 |
| SER&ACC | 1.995 |
| DESIGN | 1.378 |
| SCREEN | 949 |
| STORAGE | 91 |

Hai điểm mất cân bằng có hệ quả trực tiếp đến công bằng của mô hình và được
phân tích ở mục 2.4 của báo cáo ethics: lớp Neutral chỉ chiếm 12,2%, và aspect
STORAGE chỉ có 91 mẫu, ít hơn aspect lớn nhất khoảng 53 lần.

## 4. Rò rỉ dữ liệu

Các vectorizer TF-IDF được khớp **chỉ trên tập train** rồi mới transform tập dev
và test (`src/train_baseline.py`, hàm `build_features`). Khớp trên toàn bộ dữ
liệu sẽ đưa thông tin phân bố từ vựng của tập test vào quá trình huấn luyện và
làm kết quả báo cáo cao hơn thực tế.

Phân chia train/dev/test giữ nguyên theo bản gốc của nhóm tác giả, không chia
lại, để kết quả so sánh được với các công trình khác trên cùng bộ dữ liệu.
