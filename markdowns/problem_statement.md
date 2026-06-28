# Mô tả các bài toán trong hệ thống dự đoán và đánh giá giá laptop

## 1. Bài toán 1: Dự đoán giá laptop

### Mục tiêu

Bài toán chính của hệ thống là dự đoán mức giá hợp lý của một chiếc laptop dựa trên các thông tin cấu hình và tình trạng máy.

### Đầu vào

Đầu vào là thông tin của một laptop, bao gồm các đặc trưng như:

- Dung lượng RAM
- Dung lượng ổ cứng
- Loại ổ cứng
- Kích thước màn hình
- Thương hiệu
- Dòng máy
- Hãng CPU
- Phân cấp CPU
- Phân cấp GPU
- Tình trạng máy
- Trạng thái bảo hành
- ...

Ví dụ đầu vào:

```text
RAM: 8GB
Storage: 512GB
Storage type: SSD
Screen size: 15.6 inch
Brand: Dell
CPU: Intel Core i5
GPU: Integrated
Condition: Used
Warranty: No warranty
```

### Đầu ra

Đầu ra là mức giá hợp lý mà mô hình dự đoán cho laptop.

Ví dụ đầu ra:

```text
Predicted price: 12.000.000 VND
```

### Cách hoạt động

Hệ thống sử dụng dữ liệu laptop đã có giá bán thực tế để huấn luyện mô hình dự đoán giá. Mô hình học mối quan hệ giữa cấu hình laptop và giá bán. Sau khi huấn luyện, khi người dùng nhập thông tin một laptop mới, mô hình sẽ dự đoán mức giá phù hợp cho laptop đó.

Giá nên là giá dao động vì người dùng đôi khi sẽ không bao giờ nhập hết tất cả thông tin. Do đó, hệ thống nên đưa ra một khoảng giá dự đoán thay vì một con số cố định.

---

## 2. Bài toán 2: So sánh giá giữa nhiều cấu hình

### Mục tiêu

Bài toán này giúp người dùng hiểu việc thay đổi một hoặc nhiều thành phần cấu hình sẽ ảnh hưởng như thế nào đến giá dự đoán của laptop.

Ví dụ, người dùng có thể muốn biết:

```text
Nếu cùng một laptop nhưng RAM tăng từ 8GB lên 16GB hoặc 32GB thì giá dự đoán thay đổi như thế nào?
```

### Đầu vào

Đầu vào là một cấu hình laptop gốc và một hoặc nhiều thành phần cần thay đổi.

Ví dụ đầu vào:

```text
Cấu hình gốc:
Brand: Dell
CPU: Intel Core i5
Storage: 512GB SSD
Screen size: 15.6 inch
Condition: Used

Thành phần thay đổi:
RAM = 8GB, 16GB, 32GB
```

Hoặc:

```text
Cấu hình gốc:
RAM: 16GB
CPU: Intel Core i5
Screen size: 15.6 inch
Condition: Used

Thành phần thay đổi:
Storage = 256GB, 512GB, 1TB
```

### Đầu ra

Đầu ra là danh sách các cấu hình sau khi thay đổi và giá dự đoán tương ứng.

Ví dụ đầu ra:

```text
RAM 8GB  -> Predicted price: 10.000.000 VND
RAM 16GB -> Predicted price: 12.000.000 VND
RAM 32GB -> Predicted price: 14.000.000 VND
```

Dữ liệu này có thể được dùng để vẽ biểu đồ trên giao diện web.

Ví dụ:

```text
Trục X: Dung lượng RAM
Trục Y: Giá dự đoán
```

### Cách hoạt động

Hệ thống giữ nguyên phần lớn thông tin của laptop, chỉ thay đổi một đặc trưng cần phân tích, ví dụ RAM hoặc dung lượng ổ cứng. Với mỗi cấu hình mới, hệ thống đưa vào mô hình dự đoán giá để lấy ra giá tương ứng. Sau đó, kết quả được tổng hợp thành bảng hoặc biểu đồ để người dùng dễ so sánh.

Bài toán này không cần mô hình mới, mà sử dụng lại mô hình dự đoán giá đã huấn luyện.

---

## 3. Bài toán 3: So sánh nhiều laptop và xếp hạng mức độ đáng mua

### Mục tiêu

Bài toán này giúp người dùng so sánh nhiều laptop khác nhau và chọn ra laptop có mức giá tốt hơn so với cấu hình.

Thay vì chỉ hỏi:

```text
Laptop này giá có hợp lý không?
```

người dùng có thể hỏi:

```text
Trong các laptop này, laptop nào đáng mua nhất?
```

### Đầu vào

Đầu vào là danh sách nhiều laptop, mỗi laptop có thông tin cấu hình và giá bán thực tế.

Ví dụ đầu vào:

```text
Laptop A:
RAM: 8GB
Storage: 512GB SSD
CPU: Intel Core i5
Condition: Used
Actual price: 12.000.000 VND

Laptop B:
RAM: 16GB
Storage: 512GB SSD
CPU: Intel Core i5
Condition: Used
Actual price: 14.000.000 VND

Laptop C:
RAM: 16GB
Storage: 1TB SSD
CPU: Intel Core i7
Condition: Used
Actual price: 18.000.000 VND
```

### Đầu ra

Đầu ra là bảng so sánh gồm:

- Giá bán dự đoán cho mỗi laptop dựa trên mô hình
- Giá hợp lý do mô hình dự đoán
- Mức chênh lệch giữa giá thực tế và giá dự đoán
- Thứ hạng đáng mua

Ví dụ đầu ra:

```text
Laptop B: Đáng mua nhất
Laptop C: Giá tương đối hợp lý
Laptop A: Hơi đắt so với cấu hình
```

Hoặc dạng bảng:

| Laptop | Actual price | Predicted price | Nhận xét |
|---|---:|---:|---|
| Laptop A | 12.000.000 | 11.000.000 | Hơi đắt |
| Laptop B | 14.000.000 | 15.500.000 | Đáng mua |
| Laptop C | 18.000.000 | 18.200.000 | Hợp lý |

### Cách hoạt động

Với mỗi laptop trong danh sách, hệ thống dùng mô hình dự đoán giá để tính ra giá hợp lý. Sau đó, hệ thống so sánh giá bán thực tế với giá dự đoán.

Nếu giá bán thực tế thấp hơn giá dự đoán, laptop đó được đánh giá là có lợi hơn cho người mua. Nếu giá bán thực tế cao hơn giá dự đoán, laptop đó bị xem là kém hấp dẫn hơn.

Cuối cùng, hệ thống xếp hạng các laptop dựa trên mức chênh lệch giữa giá thực tế và giá dự đoán.

Bài toán này không cần mô hình mới, mà là một lớp so sánh và xếp hạng dựa trên kết quả của mô hình dự đoán giá.

---

## 4. Bài toán 4: Phát hiện tin đăng có dấu hiệu bất thường

### Mục tiêu

Bài toán này nhằm phát hiện các tin đăng laptop có dấu hiệu bất thường dựa trên mức độ tương xứng giữa cấu hình, giá bán thực tế và giá dự đoán của mô hình.

Thay vì khẳng định một tin đăng là giả hoặc không đáng tin, hệ thống chỉ đưa ra cảnh báo rằng tin đăng có thể cần được kiểm tra kỹ hơn. Dấu hiệu bất thường ở đây không chỉ đến từ việc giá bán lệch xa so với giá dự đoán, mà còn đến từ việc một số thành phần cấu hình như CPU, GPU, RAM hoặc dung lượng lưu trữ có thể quá thấp hoặc quá cao so với mức giá đang được rao bán.

### Đầu vào

Đầu vào là thông tin của một laptop, bao gồm cấu hình hiện tại và giá bán thực tế.

Ví dụ đầu vào:

```text
RAM: 8GB
Storage: 256GB SSD
Screen size: 15.6 inch
CPU: Intel Core i5
GPU: Integrated
Condition: Used
Actual price: 18.000.000 VND
```
Hệ thống sẽ sử dụng mô hình dự đoán giá để tính giá hợp lý cho cấu hình hiện tại.

Ví dụ:
```
Predicted price: 13.500.000 VND
```
Ngoài ra, hệ thống có thể tạo thêm một số cấu hình giả định bằng cách giữ nguyên phần lớn thông tin đã biết và chỉ thay đổi một vài thành phần quan trọng như RAM, storage, CPU hoặc GPU.

Ví dụ:
```
Giữ nguyên CPU, GPU, storage, condition.
Thay đổi RAM:
RAM 8GB  -> Predicted price: 13.500.000 VND
RAM 16GB -> Predicted price: 16.800.000 VND
RAM 32GB -> Predicted price: 19.500.000 VND
```
Đầu ra

Đầu ra là kết luận về mức độ bất thường của tin đăng.

Ví dụ:
```
Normal
```
Hoặc:
```
Suspicious
```
Có thể mở rộng thành nhiều mức:
```
Normal
Low risk
Medium risk
High risk
```
Ví dụ đầu ra chi tiết:

Risk level: Medium risk
Reason: Giá bán thực tế cao hơn khá nhiều so với giá dự đoán của cấu hình hiện tại. Với mức giá 18.000.000 VND, model cho thấy cấu hình RAM 8GB có thể hơi thấp, vì mức giá này gần với cấu hình RAM 16GB hơn.
Cách hoạt động

Bài toán này không cần huấn luyện thêm một mô hình phân loại riêng. Hệ thống sử dụng lại mô hình dự đoán giá làm lõi, sau đó thực hiện các bước kiểm tra phía sau.

Đầu tiên, hệ thống dự đoán giá hợp lý cho cấu hình hiện tại của laptop. Sau đó, hệ thống so sánh giá bán thực tế với giá dự đoán để tính độ lệch giá.

Ví dụ:
```
Actual price: 18.000.000 VND
Predicted price: 13.500.000 VND
Price gap: +33.3%
```
Nếu giá bán thực tế lệch quá xa so với giá dự đoán, tin đăng có thể được xem là có dấu hiệu bất thường.

Tuy nhiên, hệ thống không chỉ dựa vào độ lệch tổng thể. Để phân tích kỹ hơn, hệ thống giữ nguyên các thông tin đã biết của laptop và lần lượt thay đổi một số thành phần quan trọng như RAM, dung lượng lưu trữ, CPU hoặc GPU. Với mỗi cấu hình giả định, mô hình sẽ dự đoán lại giá.

Ví dụ kiểm tra RAM:
```
RAM 8GB  -> Predicted price: 13.500.000 VND
RAM 16GB -> Predicted price: 16.800.000 VND
RAM 32GB -> Predicted price: 19.500.000 VND
```
Nếu laptop thực tế chỉ có RAM 8GB nhưng giá bán lại gần với mức giá mà mô hình dự đoán cho cấu hình RAM 16GB hoặc 32GB, hệ thống có thể nhận xét rằng RAM hiện tại đang thấp so với mức giá bán.

Tương tự, hệ thống có thể kiểm tra các thành phần khác như storage, CPU hoặc GPU. Nếu nhiều thành phần quan trọng đều thấp hơn so với mức giá bán, tin đăng có thể bị đánh dấu là có dấu hiệu bất thường hoặc overpriced. Ngược lại, nếu cấu hình cao nhưng giá bán thấp bất thường so với giá dự đoán, hệ thống cũng có thể cảnh báo người dùng cần kiểm tra lại tin đăng.

Các thông tin bị thiếu không được xem trực tiếp là lý do để gắn nhãn suspicious, vì người bán hoặc người dùng có thể không cung cấp đầy đủ thông tin. Trong bài toán này, yếu tố chính để phát hiện bất thường là độ lệch giữa giá thực tế và giá dự đoán, kết hợp với việc kiểm tra mức độ tương xứng của từng thành phần cấu hình so với giá bán.

Tóm tắt

Bài toán phát hiện tin đăng bất thường được xây dựng dựa trên mô hình dự đoán giá đã có. Hệ thống không cần thêm nhãn thủ công và không cần huấn luyện thêm một mô hình classification riêng. Thay vào đó, hệ thống sử dụng mô hình dự đoán giá nhiều lần trên các cấu hình hiện tại và cấu hình giả định để đánh giá xem giá bán có hợp lý với cấu hình hay không.
```
Input: Cấu hình laptop + giá bán thực tế
Output: Normal / Suspicious hoặc mức độ rủi ro
Core idea: Dùng model dự đoán giá để kiểm tra độ lệch giá và độ tương xứng của CPU, GPU, RAM, storage so với giá b
```

Ví dụ đầu ra chi tiết:

```text
Risk level: High risk
Reason: RAM bất thường, thiếu thông tin CPU/GPU, giá bán thấp hơn rất nhiều so với giá dự đoán.
```

### Cách hoạt động

Có hai cách triển khai bài toán này.

#### Cách 1: Dùng quy tắc

Hệ thống xác định tin đăng đáng nghi dựa trên một số quy tắc có sẵn.

Ví dụ:

```text
Nếu RAM bất thường
hoặc dung lượng ổ cứng bất thường
hoặc thiếu nhiều thông tin quan trọng
hoặc giá bán lệch quá xa so với giá dự đoán
thì đánh dấu là suspicious.
```

Cách này dễ triển khai, không cần nhãn thủ công, nhưng kết quả phụ thuộc vào các quy tắc do người xây dựng hệ thống đặt ra.

#### Cách 2: Tạo nhãn tự động rồi huấn luyện mô hình phân loại

Trước tiên, hệ thống tạo nhãn `suspicious` tự động dựa trên các quy tắc bất thường. Sau đó, dùng các nhãn này để huấn luyện một mô hình phân loại.

Khi đó bài toán có dạng:

```text
Input: thông tin laptop + các dấu hiệu bất thường
Output: suspicious hoặc not suspicious
```

Cách này giống một bài toán classification hơn, nhưng cần ghi rõ rằng nhãn được tạo tự động, không phải nhãn được gán thủ công bởi con người.

---

## 5. Tổng kết hệ thống

Các bài toán trong hệ thống có quan hệ với nhau như sau:

```text
Thông tin laptop
        |
        v
Mô hình dự đoán giá
        |
        v
Giá dự đoán
        |
        |--------------------------|
        |                          |
        v                          v
So sánh nhiều cấu hình      So sánh nhiều laptop
        |                          |
        v                          v
Biểu đồ giá dự đoán         Xếp hạng mức độ đáng mua
        |
        v
Phân tích ảnh hưởng của cấu hình đến giá
```

Riêng bài toán phát hiện tin đăng đáng nghi sử dụng thêm các dấu hiệu bất thường và mức chênh lệch giữa giá thực tế với giá dự đoán:

```text
Thông tin laptop
+ dấu hiệu bất thường
+ giá thực tế
+ giá dự đoán
        |
        v
Phân loại tin đăng
        |
        v
Suspicious / Not suspicious
```

Nhìn chung, hệ thống có một bài toán chính là dự đoán giá laptop. Các bài toán còn lại là các hướng mở rộng sử dụng kết quả dự đoán giá để hỗ trợ người dùng so sánh, đánh giá và phát hiện các trường hợp bất thường.
