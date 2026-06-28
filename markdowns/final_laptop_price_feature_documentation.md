# Final Laptop Price Feature Encoding Documentation

## 1. Mục đích tài liệu

Tài liệu này mô tả **toàn bộ 86 feature cuối cùng** đang được dùng bởi production model dự đoán giá laptop. Model cuối nhận input là một `pandas.DataFrame` đã được feature engineering sẵn, không nhận trực tiếp raw input như `brand = Apple`, `cpu = Intel Core i5-1235U`, `storage = 512GB SSD`.

Thông tin quan trọng:

| Thuộc tính | Giá trị |
|---|---|
| Target | `target_price` |
| Số feature cuối | `86` |
| Input format | `pandas.DataFrame` |
| Feature order | bắt buộc đúng thứ tự schema |
| Missing feature policy | thiếu feature thì raise error |
| Extra feature policy | cột dư nên drop trước khi predict |
| Model cuối | `CatBoost_tuned` |

> Ghi chú: tài liệu này ưu tiên mô tả theo **final schema sau feature selection**. Một số feature từng được tạo ở notebook 06 nhưng đã bị notebook 11 loại bỏ, ví dụ `brand_Other`, `brand_Gigabyte`, `brand_Sony`, `brand_Toshiba`, `model_Other`, `cpu_brand_Other`, `cpu_brand_Microsoft SQ`, `cpu_qualcomm_snapdragon_spec`, `gpu_type_*`.

---

## 2. Nguyên tắc encode tổng quát

Pipeline cuối có các nguyên tắc sau:

1. **Numeric feature giữ dạng số**  
   Ví dụ: `ram_gb`, `storage_gb`, `screen_size_inch`.

2. **Categorical không có thứ tự dùng one-hot encoding**  
   Ví dụ: brand, model, CPU brand, CPU family group.  
   Nếu category có trong final schema thì cột tương ứng bằng `1`, các category khác bằng `0`.

3. **Categorical có thứ tự dùng ordinal encoding**  
   Ví dụ: `condition_score`, `cpu_tier_encoded`, `cpu_family_ord_filled`, `gpu_tier_ord_filled`, `cpu_suffix_power_ord_filled`.

4. **Missing thật sự dùng `NaN` ở raw input và tạo missing/no-info flag**  
   Với input mới, không cần tự tạo chuỗi `Other`, `Unknown`, `orther`. Nếu không có giá trị thì để `NaN`, sau đó pipeline tạo các flag như `no_info_brand`, `no_info_model`, `ram_missing`, `no_info_gpu`.

5. **Final schema chỉ giữ các cột đã được chọn**  
   Brand/model/CPU/GPU nào không còn nằm trong final schema thì không tạo cột riêng. Nếu vẫn cần giữ tín hiệu hiếm, dùng `brand_is_rare`, `model_is_rare`, hoặc segment flag tương ứng.

---

## 3. Danh sách feature cuối cùng theo đúng thứ tự

```text
01. ram_gb
02. storage_gb
03. screen_size_inch
04. brand_is_rare
05. model_is_rare
06. ram_missing
07. storage_missing
08. screen_missing
09. no_info_brand
10. no_info_model
11. no_info_cpu_brand
12. no_info_cpu_tier
13. no_info_gpu
14. storage_ssd
15. storage_hdd
16. no_info_storage
17. condition_score
18. cpu_tier_encoded
19. brand_ASUS
20. brand_Acer
21. brand_Apple
22. brand_Dell
23. brand_HP
24. brand_LG
25. brand_Lenovo
26. brand_MSI
27. brand_Microsoft
28. model_Aspire
29. model_Elitebook
30. model_Elitebook 800
31. model_Gaming Thin GF
32. model_IdeaPad
33. model_Inspiron
34. model_Latitude
35. model_Latitude 14 7000
36. model_Latitude E Series
37. model_Legion
38. model_Legion 5
39. model_MacBook Air
40. model_MacBook Air M1
41. model_MacBook Air M2
42. model_MacBook Pro
43. model_MacBook Pro M1
44. model_MacBook Pro M2
45. model_Macbook air m4
46. model_Nitro 5
47. model_Pavilion 15
48. model_Precision
49. model_ProBook
50. model_ROG Strix
51. model_TUF Gaming
52. model_TUF Gaming F15
53. model_ThinkPad
54. model_ThinkPad X1 Carbon
55. model_Vivobook 15
56. model_Vostro
57. model_X Series
58. model_XPS 13
59. cpu_brand_AMD
60. cpu_brand_Apple
61. cpu_brand_Intel
62. cpu_brand_Qualcomm
63. cpu_family_ord_filled
64. cpu_family_group_AMD Ryzen
65. cpu_family_group_Apple Silicon
66. cpu_family_group_Intel Core Ultra
67. cpu_family_group_Intel Core i
68. cpu_family_group_Intel Low End
69. cpu_intel_generation_ord
70. cpu_amd_generation_ord
71. cpu_apple_core_spec
72. cpu_suffix_power_ord_filled
73. gpu_tier_ord_filled
74. warranty_expired
75. warranty_active
76. warranty_not_activated
77. ram_storage_product_scaled
78. ram_storage_balance
79. memory_storage_score
80. is_entry_memory_storage
81. is_mid_memory_storage
82. is_premium_memory_storage
83. brand_segment_premium
84. brand_segment_business
85. brand_segment_gaming_value
86. brand_segment_rare
```

---

## 4. Nhóm numeric hardware features

### Feature trong nhóm

```text
ram_gb
storage_gb
screen_size_inch
```

### Raw input cần có

```text
RAM: ví dụ 8GB, 16GB, 32GB
Storage: ví dụ 256GB, 512GB, 1TB
Screen size: ví dụ 13.3, 14, 15.6 inch
```

### Cách encode

| Feature | Cách tạo |
|---|---|
| `ram_gb` | parse RAM về số GB |
| `storage_gb` | parse storage về số GB; 1TB thường quy đổi thành 1024GB |
| `screen_size_inch` | parse kích thước màn hình về số inch |

### Ví dụ

Raw input:

```text
RAM = 16GB
Storage = 512GB
Screen = 14 inch
```

Encoded:

```text
ram_gb = 16
storage_gb = 512
screen_size_inch = 14
```

---

## 5. Nhóm missing và no-info flags

### Feature trong nhóm

```text
ram_missing
storage_missing
screen_missing
no_info_brand
no_info_model
no_info_cpu_brand
no_info_cpu_tier
no_info_gpu
no_info_storage
```

### Cách encode

| Feature | Giá trị `1` khi | Giá trị `0` khi |
|---|---|---|
| `ram_missing` | raw RAM là `NaN` hoặc không parse được | có RAM hợp lệ |
| `storage_missing` | raw storage là `NaN` hoặc không parse được | có storage hợp lệ |
| `screen_missing` | raw screen size là `NaN` hoặc không parse được | có screen hợp lệ |
| `no_info_brand` | raw brand là `NaN` | có brand hợp lệ |
| `no_info_model` | raw model là `NaN` | có model hợp lệ |
| `no_info_cpu_brand` | không parse được CPU brand | parse được CPU brand |
| `no_info_cpu_tier` | không xác định được CPU tier | xác định được CPU tier |
| `no_info_gpu` | không có thông tin GPU | có thông tin GPU |
| `no_info_storage` | không xác định được loại storage | xác định được SSD/HDD |

### Quy ước input mới

Không cần đưa `Other`, `Unknown`, `orther` vào input. Nếu thiếu thông tin thì để `NaN`.

Ví dụ:

```text
brand = NaN
model = NaN
cpu = NaN
gpu = NaN
```

Encoded:

```text
no_info_brand = 1
no_info_model = 1
no_info_cpu_brand = 1
no_info_cpu_tier = 1
no_info_gpu = 1
```

---

## 6. Nhóm brand features

### Feature trong nhóm

Final schema chỉ giữ các brand one-hot sau:

```text
brand_ASUS
brand_Acer
brand_Apple
brand_Dell
brand_HP
brand_LG
brand_Lenovo
brand_MSI
brand_Microsoft
```

Ngoài ra còn có các feature liên quan:

```text
brand_is_rare
brand_segment_premium
brand_segment_business
brand_segment_gaming_value
brand_segment_rare
no_info_brand
```

### Brand đã bị loại sau feature selection

Các brand one-hot sau từng có trong dataset sau notebook 06 nhưng đã bị notebook 11 drop khỏi final schema:

```text
brand_Gigabyte
brand_Other
brand_Sony
brand_Toshiba
```

Vì vậy, khi predict production, **không tạo các cột này nữa**.

### Cách encode one-hot brand

Nếu raw brand nằm trong danh sách brand được giữ, set đúng một cột brand bằng `1`.

Ví dụ:

```text
brand = Apple
```

Encoded:

```text
brand_Apple = 1
brand_ASUS = 0
brand_Acer = 0
brand_Dell = 0
brand_HP = 0
brand_LG = 0
brand_Lenovo = 0
brand_MSI = 0
brand_Microsoft = 0
```

Nếu raw brand không nằm trong các cột brand final, không tạo cột riêng cho brand đó. Các cột brand final đều bằng `0`.

Ví dụ:

```text
brand = Sony
```

Encoded theo final schema:

```text
brand_ASUS = 0
brand_Acer = 0
brand_Apple = 0
brand_Dell = 0
brand_HP = 0
brand_LG = 0
brand_Lenovo = 0
brand_MSI = 0
brand_Microsoft = 0
```

### Cách encode `brand_is_rare`

Notebook 06 tạo `brand_is_rare` theo rule:

```python
brand_freq = df["brand_grouped"].value_counts(normalize=True, dropna=False)
df["brand_is_rare"] = (df["brand_grouped"].map(brand_freq) < 0.01).astype(int)
```

Rare brand được notebook 06 ghi nhận:

| Brand | Count | Percent |
|---|---:|---:|
| Sony | 30 | 0.397% |
| Toshiba | 26 | 0.344% |
| LG | 25 | 0.330% |
| Gigabyte | 14 | 0.185% |

Quy tắc inference nên dùng:

| Trường hợp raw brand | `brand_is_rare` | Ghi chú |
|---|---:|---|
| `Sony`, `Toshiba`, `LG`, `Gigabyte` | 1 | Theo rare list notebook 06 |
| Brand không nằm trong danh sách one-hot final và không phải `NaN` | 1 | Fallback hợp lý cho unseen/rare brand |
| Brand nằm trong one-hot final và không thuộc rare list | 0 | Brand phổ biến |
| Brand là `NaN` | 0 hoặc theo pipeline hiện tại | Tín hiệu thiếu đã nằm ở `no_info_brand = 1`; không nên đồng thời coi là rare nếu mục tiêu là tách missing khỏi rare |

> Lưu ý: `LG` là trường hợp đặc biệt: vừa có `brand_LG` trong final schema, vừa nằm trong rare list notebook 06. Vì vậy với `brand = LG`, có thể encode `brand_LG = 1` và `brand_is_rare = 1`.

### Cách encode brand segment flags

Notebook 11 tạo brand segment flags từ các one-hot brand trước khi drop một số brand sparse.

| Segment feature | Rule |
|---|---|
| `brand_segment_premium` | `brand_Apple` hoặc `brand_Microsoft` hoặc `brand_LG` |
| `brand_segment_business` | `brand_Dell` hoặc `brand_HP` hoặc `brand_Lenovo` |
| `brand_segment_gaming_value` | `brand_ASUS` hoặc `brand_Acer` hoặc `brand_MSI` hoặc `brand_Gigabyte` |
| `brand_segment_rare` | `brand_is_rare` hoặc brand thuộc nhóm rare/dropped như `Sony`, `Toshiba`, `Gigabyte` |

Ví dụ:

```text
brand = Apple
```

Encoded:

```text
brand_Apple = 1
brand_segment_premium = 1
brand_segment_business = 0
brand_segment_gaming_value = 0
brand_segment_rare = 0
```

Ví dụ:

```text
brand = Gigabyte
```

Encoded final:

```text
brand_Gigabyte không được tạo
brand_is_rare = 1
brand_segment_gaming_value = 1
brand_segment_rare = 1
```

Ví dụ:

```text
brand = Sony
```

Encoded final:

```text
brand_Sony không được tạo
brand_is_rare = 1
brand_segment_rare = 1
```

---

## 7. Nhóm model features

### Feature trong nhóm

Final schema giữ các model one-hot sau:

```text
model_is_rare
model_Aspire
model_Elitebook
model_Elitebook 800
model_Gaming Thin GF
model_IdeaPad
model_Inspiron
model_Latitude
model_Latitude 14 7000
model_Latitude E Series
model_Legion
model_Legion 5
model_MacBook Air
model_MacBook Air M1
model_MacBook Air M2
model_MacBook Pro
model_MacBook Pro M1
model_MacBook Pro M2
model_Macbook air m4
model_Nitro 5
model_Pavilion 15
model_Precision
model_ProBook
model_ROG Strix
model_TUF Gaming
model_TUF Gaming F15
model_ThinkPad
model_ThinkPad X1 Carbon
model_Vivobook 15
model_Vostro
model_X Series
model_XPS 13
```

Ngoài ra còn có:

```text
model_is_rare
no_info_model
```

### Model đã bị loại sau feature selection

Notebook 11 quyết định drop:

```text
model_Other
```

Vì vậy production input **không cần tạo `model_Other`**. Nếu model không nằm trong danh sách cột final, các model one-hot đều bằng `0`, còn tín hiệu hiếm được giữ bằng `model_is_rare` nếu phù hợp.

### Cách encode one-hot model

Nếu raw model nằm trong danh sách model final, set đúng cột model tương ứng bằng `1`.

Ví dụ:

```text
model = MacBook Air M2
```

Encoded:

```text
model_MacBook Air M2 = 1
các model_* khác = 0
```

Nếu raw model không nằm trong final model columns:

```text
model = Zephyrus G14
```

Encoded:

```text
không tạo model_Zephyrus G14
mọi model_* trong final schema = 0
model_is_rare = 1  # fallback hợp lý nếu model không thuộc danh sách final và không phải NaN
```

### Cách encode `model_is_rare`

Notebook 06 tạo `model_is_rare` theo rule:

```python
model_freq = df["model_grouped"].value_counts(normalize=True, dropna=False)
df["model_is_rare"] = (df["model_grouped"].map(model_freq) < 0.01).astype(int)
```

Rare models được notebook 06 ghi nhận gồm:

| Model | Count | Percent |
|---|---:|---:|
| Vostro | 75 | 0.991% |
| ThinkPad X1 Carbon | 65 | 0.859% |
| Legion 5 | 59 | 0.780% |
| Latitude 14 7000 | 58 | 0.767% |
| Macbook air m4 | 51 | 0.674% |
| Latitude E Series | 49 | 0.648% |
| TUF Gaming F15 | 42 | 0.555% |
| XPS 13 | 42 | 0.555% |
| TUF Gaming | 42 | 0.555% |
| MacBook Air M2 | 41 | 0.542% |
| Pavilion 15 | 40 | 0.529% |
| Vivobook 15 | 40 | 0.529% |
| Gaming Thin GF | 40 | 0.529% |
| Elitebook 800 | 37 | 0.489% |
| X Series | 36 | 0.476% |
| IdeaPad | 33 | 0.436% |
| MacBook Pro M2 | 30 | 0.397% |
| Aspire | 30 | 0.397% |
| ROG Strix | 29 | 0.383% |
| Legion | 26 | 0.344% |

Điểm cần chú ý: nhiều model hiếm ở notebook 06 vẫn được giữ one-hot trong final schema. Ví dụ `model_MacBook Air M2`, `model_ROG Strix`, `model_ThinkPad X1 Carbon`. Do đó một model có thể vừa có one-hot riêng, vừa có `model_is_rare = 1`.

Ví dụ:

```text
model = ThinkPad X1 Carbon
```

Encoded:

```text
model_ThinkPad X1 Carbon = 1
model_is_rare = 1
```

Ví dụ:

```text
model = Inspiron
```

Encoded:

```text
model_Inspiron = 1
model_is_rare = 0  # nếu Inspiron không thuộc rare list notebook 06
```

Ví dụ missing:

```text
model = NaN
```

Encoded:

```text
no_info_model = 1
mọi model_* = 0
model_is_rare = 0  # nên tách missing khỏi rare
```

---

## 8. Nhóm storage type features

### Feature trong nhóm

```text
storage_ssd
storage_hdd
no_info_storage
```

### Cách encode

| Raw storage type | `storage_ssd` | `storage_hdd` | `no_info_storage` |
|---|---:|---:|---:|
| SSD | 1 | 0 | 0 |
| HDD | 0 | 1 | 0 |
| SSD + HDD hoặc hybrid | 1 | 1 | 0 |
| `NaN` / không parse được | 0 | 0 | 1 |

Ví dụ:

```text
storage = 512GB SSD
```

Encoded:

```text
storage_gb = 512
storage_ssd = 1
storage_hdd = 0
no_info_storage = 0
```

---

## 9. Nhóm condition và warranty

### Feature trong nhóm

```text
condition_score
warranty_expired
warranty_active
warranty_not_activated
```

### Encode condition

Notebook 06 dùng ordinal encoding:

| Raw `condition_clean` | `condition_score` |
|---|---:|
| `Đã sử dụng (qua sửa chữa)` | 1 |
| `Đã sử dụng (chưa sửa chữa)` | 2 |
| `Mới` | 3 |
| Missing/unmapped | 2 |

Ví dụ:

```text
condition = Mới
```

Encoded:

```text
condition_score = 3
```

### Encode warranty

Notebook 06 tạo `warranty_encoded`, sau đó notebook 11 quyết định thay bằng one-hot warranty flags.

Mapping ban đầu:

| Raw `warranty_status` | `warranty_encoded` |
|---|---:|
| `Expired` | 0 |
| `Active` | 1 |
| `Manufacturer` | 1 |
| `not_active` | 2 |

Final one-hot:

| `warranty_encoded` | `warranty_expired` | `warranty_active` | `warranty_not_activated` |
|---:|---:|---:|---:|
| 0 | 1 | 0 | 0 |
| 1 | 0 | 1 | 0 |
| 2 | 0 | 0 | 1 |

Ví dụ:

```text
warranty_status = Active
```

Encoded:

```text
warranty_expired = 0
warranty_active = 1
warranty_not_activated = 0
```

---

## 10. Nhóm CPU features

### Feature trong nhóm

```text
cpu_tier_encoded
cpu_brand_AMD
cpu_brand_Apple
cpu_brand_Intel
cpu_brand_Qualcomm
cpu_family_ord_filled
cpu_family_group_AMD Ryzen
cpu_family_group_Apple Silicon
cpu_family_group_Intel Core Ultra
cpu_family_group_Intel Core i
cpu_family_group_Intel Low End
cpu_intel_generation_ord
cpu_amd_generation_ord
cpu_apple_core_spec
cpu_suffix_power_ord_filled
no_info_cpu_brand
no_info_cpu_tier
```

### CPU features đã bị loại sau feature selection

Notebook 11 drop các CPU columns sau:

```text
cpu_brand_Other
cpu_brand_Microsoft SQ
cpu_qualcomm_snapdragon_spec
```

Vì vậy final schema không còn các feature này.

### 10.1 CPU brand one-hot

Cách parse từ raw CPU:

| Raw CPU chứa | CPU brand |
|---|---|
| `Intel` | Intel |
| `AMD`, `Ryzen` | AMD |
| `Apple`, `M1`, `M2`, `M3`, `M4`, `M5` | Apple |
| `Qualcomm`, `Snapdragon` | Qualcomm |
| Missing/unparsed | no-info |

One-hot final:

| CPU brand | `cpu_brand_Intel` | `cpu_brand_AMD` | `cpu_brand_Apple` | `cpu_brand_Qualcomm` |
|---|---:|---:|---:|---:|
| Intel | 1 | 0 | 0 | 0 |
| AMD | 0 | 1 | 0 | 0 |
| Apple | 0 | 0 | 1 | 0 |
| Qualcomm | 0 | 0 | 0 | 1 |
| Missing/unparsed | 0 | 0 | 0 | 0 |

### 10.2 CPU family group one-hot

Notebook 06 group CPU family như sau:

| Raw CPU family | `cpu_family_group` |
|---|---|
| Intel Core i3 / i5 / i7 / i9 | Intel Core i |
| Intel Core Ultra 5 / 7 / 9 | Intel Core Ultra |
| Intel Pentium / N-Series / Celeron | Intel Low End |
| AMD Ryzen 3 / 5 / 7 / 9 / Ryzen AI | AMD Ryzen |
| Apple M1 / M2 / M3 / M4 / M5 | Apple Silicon |
| Other/unparsed | Unknown, nhưng final schema không giữ `cpu_family_group_Unknown` |

Final one-hot:

```text
cpu_family_group_AMD Ryzen
cpu_family_group_Apple Silicon
cpu_family_group_Intel Core Ultra
cpu_family_group_Intel Core i
cpu_family_group_Intel Low End
```

### 10.3 CPU family ordinal

Notebook 06 dùng `cpu_family_ord_filled`:

| Raw CPU family | `cpu_family_ord_filled` |
|---|---:|
| Other/unparsed | -1 |
| Intel Pentium / Intel N-Series / Intel Celeron | 0 |
| Intel Core i3 / AMD Ryzen 3 | 1 |
| Intel Core i5 / Intel Core Ultra 5 / AMD Ryzen 5 / Apple M1 | 2 |
| Intel Core i7 / Intel Core Ultra 7 / AMD Ryzen 7 / AMD Ryzen AI / Apple M2 / Apple M3 | 3 |
| Intel Core i9 / Intel Core Ultra 9 / AMD Ryzen 9 / Apple M4 / Apple M5 | 4 |

### 10.4 CPU generation features

Notebook 06 trích thông tin từ `cpu_model_code_group`.

#### `cpu_intel_generation_ord`

| Pattern | Output |
|---|---:|
| `Intel Gen 12 / U` | 12 |
| `Intel Gen 13 / H` | 13 |
| `Intel Gen 14 / HX` | 14 |
| `Intel Core Ultra 100-Series / H` | 14 |
| `Intel Core Ultra 200-Series / H` | 15 |
| `Intel Core Ultra 300-Series / H` | 16 |
| `Intel Core 100-Series / U` | 14 |
| `Intel N-Series / NoSuffix` | 1 |
| Missing/non-Intel | 0 |

#### `cpu_amd_generation_ord`

| Pattern | Output |
|---|---:|
| `AMD Ryzen 2000 / U` | 2 |
| `AMD Ryzen 3000 / H` | 3 |
| `AMD Ryzen 5000 / U` | 5 |
| `AMD Ryzen 7000 / HS` | 7 |
| `AMD Ryzen 9000 / NoSuffix` | 9 |
| `AMD Ryzen AI ...` | 9 |
| Missing/non-AMD | 0 |

#### `cpu_apple_core_spec`

| Pattern | Output |
|---|---:|
| `Apple Core Count Spec` | 1 |
| Không có Apple core count spec | 0 |

### 10.5 CPU suffix power ordinal

Notebook 06 tách suffix sau dấu `/` trong `cpu_model_code_group`.

| Suffix | `cpu_suffix_power_ord_filled` | Ý nghĩa tương đối |
|---|---:|---|
| `NoSuffix`, `Y`, `M`, `F`, `D` | 0 | không suffix / rất tiết kiệm điện / không rõ |
| `G`, `G1`, `G4`, `G7`, `U`, `V`, `UC` | 1 | low power |
| `P` | 2 | mid power |
| `H`, `HS`, `HQ`, `MQ`, `HH` | 3 | high performance |
| `HX`, `HK`, `K` | 4 | very high performance / unlocked |

### 10.6 CPU tier encoded

`cpu_tier_encoded` là feature biểu diễn phân khúc hiệu năng CPU. Sau khi bổ sung bảng audit CPU tier, cách mô tả nên dùng là:

```text
cpu_tier_encoded = numeric code của cpu_tier
cpu_tier = kết quả phân loại từ tổ hợp CPU
```

Không nên chỉ dùng một cột duy nhất như `cpu_family_from_raw` để suy ra tier, vì cùng là `Intel Core i5`, `Intel Core i7`, `Ryzen 5`, `Ryzen 7` nhưng generation và suffix như `U`, `P`, `H`, `HS`, `HX` có thể làm phân khúc hiệu năng khác nhau. Rule tốt hơn là xét tổ hợp:

```text
cpu_brand + cpu_family_from_raw + cpu_model_code_group
```

Trong đó:

| Thành phần | Ví dụ | Vai trò |
|---|---|---|
| `cpu_brand` | `Intel`, `AMD`, `Apple`, `Qualcomm` | Xác định hệ sinh thái CPU |
| `cpu_family_from_raw` | `Intel Core i5`, `Intel Core i7`, `AMD Ryzen 5`, `Apple M2` | Xác định dòng CPU chính |
| `cpu_model_code_group` | `Intel Gen 12 / U`, `Intel Gen 13 / HX`, `AMD Ryzen 7000 / HS` | Xác định generation và suffix hiệu năng |

#### Mapping numeric của CPU tier

Theo bảng audit hiện tại, `cpu_tier_numeric` đang map như sau:

| `cpu_tier_encoded` | `cpu_tier` | Ý nghĩa tương đối |
|---:|---|---|
| 0 | `Upper-mid` | cận cao cấp |
| 1 | `Entry` | nhập môn / phổ thông thấp |
| 2 | `Other` | không phân loại rõ hoặc tổ hợp không đủ tin cậy |
| 3 | `Mid` | trung bình |
| 4 | `High` | cao |
| 5 | `Low` | thấp |
| 6 | `Mid-range` | tầm trung |
| 7 | `High-end` | cao cấp |
| 8 | `Low-end` | rất thấp / low-end |

> Lưu ý: numeric code này là mã hóa category, **không phải thang ordinal tăng dần theo hiệu năng**. Ví dụ `Upper-mid = 0` nhưng không có nghĩa là yếu hơn `Entry = 1`. Khi giải thích model, nên giải thích theo label tier thay vì so sánh trực tiếp giá trị số.

#### Rule encode chính khi đủ thông tin

Khi parse được đủ 3 thành phần CPU, dùng lookup theo tổ hợp:

```text
(cpu_brand, cpu_family_from_raw, cpu_model_code_group) -> cpu_tier -> cpu_tier_encoded
```

Ví dụ minh họa:

| Raw CPU | `cpu_brand` | `cpu_family_from_raw` | `cpu_model_code_group` | `cpu_tier` | `cpu_tier_encoded` |
|---|---|---|---|---|---:|
| Intel Core i5-1235U | Intel | Intel Core i5 | Intel Gen 12 / U | Mid-range | 6 |
| Intel Core i7-1255U | Intel | Intel Core i7 | Intel Gen 12 / U | Upper-mid | 0 |
| Intel Core i9-13900HX | Intel | Intel Core i9 | Intel Gen 13 / HX | High-end | 7 |
| Intel Celeron N4020 | Intel | Intel Celeron | Intel N-Series / NoSuffix | Low-end | 8 |
| AMD Ryzen 5 5500U | AMD | AMD Ryzen 5 | AMD Ryzen 5000 / U | Mid-range | 6 |
| AMD Ryzen 7 5800H | AMD | AMD Ryzen 7 | AMD Ryzen 5000 / H | Upper-mid | 0 |
| AMD Ryzen 9 7945HX | AMD | AMD Ryzen 9 | AMD Ryzen 7000 / HX | High-end | 7 |

#### Trường hợp tổ hợp bị trùng nhiều tier

Bảng audit có một số tổ hợp thô như `Other / Other` hoặc family có `cpu_model_code_group = Other` xuất hiện ở nhiều tier khác nhau. Vì vậy, nếu xây lookup tự động từ dữ liệu, nên xử lý như sau:

1. Ưu tiên tổ hợp có `cpu_model_code_group` cụ thể, ví dụ `Intel Gen 12 / U`, `AMD Ryzen 5000 / H`, `Intel Gen 13 / HX`.
2. Với tổ hợp quá chung như `Intel + Other + Other`, `AMD + Other + Other`, không nên gán tier mạnh tay.
3. Nếu một tổ hợp xuất hiện ở nhiều tier, chọn tier có `frequency` cao nhất hoặc đưa vào nhóm cần audit thủ công.
4. Nếu không chắc, fallback về `Other` và bật `no_info_cpu_tier = 1`.

#### Rule fallback khi không đủ thông tin

Production input không cần nhập chuỗi `Other` hoặc `Unknown`. Nếu không có thông tin thì để `NaN`, sau đó pipeline tạo flag thiếu thông tin. Tuy nhiên vì model vẫn yêu cầu đủ 86 feature, các feature numeric vẫn cần giá trị fallback.

| Trường hợp raw CPU | Cách xử lý |
|---|---|
| CPU hoàn toàn thiếu / `NaN` | `no_info_cpu_brand = 1`, `no_info_cpu_tier = 1`, các one-hot CPU brand/family = 0, `cpu_tier_encoded = 2` (`Other`) |
| Có brand nhưng không có family | set brand one-hot nếu brand nằm trong schema, `no_info_cpu_tier = 1`, `cpu_tier_encoded = 2` |
| Có family nhưng không có generation/suffix | dùng family-level fallback nếu rule đủ tin cậy; nếu không, `cpu_tier_encoded = 2` và `no_info_cpu_tier = 1` |
| Có đủ 3 thành phần nhưng tổ hợp không có trong lookup | fallback `cpu_tier_encoded = 2`, `no_info_cpu_tier = 1` hoặc audit thêm lookup mới |
| Có đủ 3 thành phần và match lookup | set đúng `cpu_tier_encoded`, `no_info_cpu_tier = 0` |

Ví dụ CPU thiếu hoàn toàn:

```text
raw_cpu = NaN
```

Encoded:

```text
cpu_brand_AMD = 0
cpu_brand_Apple = 0
cpu_brand_Intel = 0
cpu_brand_Qualcomm = 0

cpu_family_group_AMD Ryzen = 0
cpu_family_group_Apple Silicon = 0
cpu_family_group_Intel Core Ultra = 0
cpu_family_group_Intel Core i = 0
cpu_family_group_Intel Low End = 0

cpu_family_ord_filled = 0
cpu_intel_generation_ord = 0
cpu_amd_generation_ord = 0
cpu_apple_core_spec = 0
cpu_suffix_power_ord_filled = 0

cpu_tier_encoded = 2
no_info_cpu_brand = 1
no_info_cpu_tier = 1
```

Ví dụ chỉ có brand:

```text
raw_cpu = Intel, nhưng không parse được i3/i5/i7, generation, suffix
```

Encoded:

```text
cpu_brand_Intel = 1
cpu_brand_AMD = 0
cpu_brand_Apple = 0
cpu_brand_Qualcomm = 0

cpu_family_group_* = 0
cpu_family_ord_filled = 0
cpu_intel_generation_ord = 0
cpu_suffix_power_ord_filled = 0

cpu_tier_encoded = 2
no_info_cpu_brand = 0
no_info_cpu_tier = 1
```

Ví dụ có family nhưng thiếu model code:

```text
raw_cpu = Intel Core i5, nhưng không biết generation/suffix
```

Encoded an toàn:

```text
cpu_brand_Intel = 1
cpu_family_group_Intel Core i = 1
cpu_family_ord_filled = 2
cpu_intel_generation_ord = 0
cpu_suffix_power_ord_filled = 0

cpu_tier_encoded = 2
no_info_cpu_brand = 0
no_info_cpu_tier = 1
```

Nếu sau này bạn quyết định dùng family-level fallback, có thể map `Intel Core i5` thiếu generation về `Mid` hoặc `Mid-range`, nhưng bản production documentation nên ghi rõ đây là fallback không chắc chắn.

### 10.7 Ví dụ encode CPU: Intel Core i5-1235U

Giả sử parser đưa raw CPU về:

```text
cpu_brand = Intel
cpu_family_from_raw = Intel Core i5
cpu_model_code_group = Intel Gen 12 / U
cpu_tier = Mid-range
```

Encoded final:

```text
cpu_brand_Intel = 1
cpu_brand_AMD = 0
cpu_brand_Apple = 0
cpu_brand_Qualcomm = 0

cpu_family_group_Intel Core i = 1
cpu_family_group_Intel Core Ultra = 0
cpu_family_group_Intel Low End = 0
cpu_family_group_AMD Ryzen = 0
cpu_family_group_Apple Silicon = 0

cpu_family_ord_filled = 2        # Intel Core i5
cpu_intel_generation_ord = 12    # Intel Gen 12
cpu_amd_generation_ord = 0
cpu_apple_core_spec = 0
cpu_suffix_power_ord_filled = 1  # U suffix

cpu_tier_encoded = 6              # Mid-range
no_info_cpu_brand = 0
no_info_cpu_tier = 0
```

### 10.8 Ví dụ encode CPU: Intel Core i7-12700H

```text
cpu_brand = Intel
cpu_family_from_raw = Intel Core i7
cpu_model_code_group = Intel Gen 12 / H
```

Encoded:

```text
cpu_brand_Intel = 1
cpu_family_group_Intel Core i = 1
cpu_family_ord_filled = 3
cpu_intel_generation_ord = 12
cpu_suffix_power_ord_filled = 3
cpu_amd_generation_ord = 0
cpu_apple_core_spec = 0

cpu_tier_encoded = 0              # Upper-mid
no_info_cpu_brand = 0
no_info_cpu_tier = 0
```

### 10.9 Ví dụ encode CPU: Apple M2

```text
cpu_brand = Apple
cpu_family_from_raw = Apple M2
cpu_model_code_group = Apple Core Count Spec hoặc Other tùy parser
```

Encoded:

```text
cpu_brand_Apple = 1
cpu_family_group_Apple Silicon = 1
cpu_family_ord_filled = 3
cpu_intel_generation_ord = 0
cpu_amd_generation_ord = 0
cpu_apple_core_spec = 1 nếu có Apple Core Count Spec, ngược lại 0
cpu_suffix_power_ord_filled = 0
```

---

## 11. Nhóm GPU features

### Feature trong nhóm final

```text
gpu_tier_ord_filled
no_info_gpu
```

Notebook 06 từng tạo thêm các cột `gpu_type_*`, nhưng notebook 11 quyết định drop toàn bộ GPU type one-hot:

```text
gpu_type_Apple SoC
gpu_type_Dedicated
gpu_type_Integrated
gpu_type_Missing_Info
```

### Cách tạo GPU tier trước khi final selection

Notebook 06 kết hợp `gpu_tier` và `gpu_tier_v2`:

1. Ưu tiên `gpu_tier_v2` nếu có thông tin cụ thể.
2. Nếu `gpu_tier_v2` missing/Other thì fallback về `gpu_tier`.
3. Sau đó chuẩn hóa về `gpu_tier_final_clean`.
4. Map sang `gpu_tier_clean`.
5. Ordinal encode thành `gpu_tier_ord_filled`.

### GPU tier ordinal map

| `gpu_tier_clean` | `gpu_tier_ord_filled` |
|---|---:|
| Unknown | -1 |
| Integrated | 0 |
| Apple_SoC | 1 |
| Entry | 2 |
| Mid_High | 3 |
| High_Workstation | 4 |
| Very_High_Workstation | 5 |

### Mapping ví dụ

| Raw GPU group | Clean group | Ordinal |
|---|---|---:|
| Intel Integrated | Integrated | 0 |
| AMD Integrated / AMD Radeon | Integrated | 0 |
| Apple GPU | Apple_SoC | 1 |
| GTX / Dedicated Entry / Other GPU | Entry | 2 |
| RTX Other | Mid_High | 3 |
| RTX 4000 | High_Workstation | 4 |
| RTX 5000 | Very_High_Workstation | 5 |
| Missing / Other | Unknown | -1 |

Ví dụ:

```text
gpu = RTX 4000
```

Encoded final:

```text
gpu_tier_ord_filled = 4
no_info_gpu = 0
```

Ví dụ:

```text
gpu = NaN
```

Encoded final:

```text
gpu_tier_ord_filled = -1
no_info_gpu = 1
```

---

## 12. Nhóm RAM / Storage interaction features

### Feature trong nhóm

```text
ram_storage_product_scaled
ram_storage_balance
memory_storage_score
is_entry_memory_storage
is_mid_memory_storage
is_premium_memory_storage
```

Notebook 11 thêm nhóm này vì experiment `rs2_ram_storage_interactions` là nhóm RAM/storage tốt nhất.

### Cách encode

Với:

```python
ram = ram_gb
storage = storage_gb
```

Công thức:

```python
ram_storage_product_scaled = (ram * storage) / (16 * 512)
ram_storage_balance = min(ram / 16, storage / 512)
memory_storage_score = 2.0 * log1p(ram) + log1p(storage)
is_entry_memory_storage = int((ram <= 8) and (storage <= 256))
is_mid_memory_storage = int((ram >= 16) and (storage >= 512))
is_premium_memory_storage = int((ram >= 32) and (storage >= 1024))
```

### Ví dụ

Raw input:

```text
ram_gb = 16
storage_gb = 512
```

Encoded:

```text
ram_storage_product_scaled = 1.0
ram_storage_balance = 1.0
is_entry_memory_storage = 0
is_mid_memory_storage = 1
is_premium_memory_storage = 0
```

Raw input:

```text
ram_gb = 32
storage_gb = 1024
```

Encoded:

```text
ram_storage_product_scaled = 4.0
ram_storage_balance = 2.0
is_entry_memory_storage = 0
is_mid_memory_storage = 1
is_premium_memory_storage = 1
```

---

## 13. Minh họa full input: Apple MacBook Air M2

Raw input:

```text
brand = Apple
model = MacBook Air M2
ram = 16GB
storage = 512GB SSD
screen = 13.6 inch
cpu = Apple M2
gpu = Apple GPU
condition = Mới
warranty_status = Active
```

Encoded các phần chính:

```text
ram_gb = 16
storage_gb = 512
screen_size_inch = 13.6

storage_ssd = 1
storage_hdd = 0
no_info_storage = 0

brand_Apple = 1
brand_segment_premium = 1
brand_is_rare = 0

model_MacBook Air M2 = 1
model_is_rare = 1

cpu_brand_Apple = 1
cpu_family_group_Apple Silicon = 1
cpu_family_ord_filled = 3
cpu_intel_generation_ord = 0
cpu_amd_generation_ord = 0
cpu_apple_core_spec = tùy parser
cpu_suffix_power_ord_filled = 0

gpu_tier_ord_filled = 1
no_info_gpu = 0

condition_score = 3
warranty_active = 1
warranty_expired = 0
warranty_not_activated = 0

ram_storage_product_scaled = 1.0
ram_storage_balance = 1.0
is_mid_memory_storage = 1
```

---

## 14. Minh họa full input: Intel Core i5-1235U laptop

Raw input:

```text
brand = Dell
model = Inspiron
ram = 16GB
storage = 512GB SSD
screen = 15.6 inch
cpu = Intel Core i5-1235U
gpu = Intel Integrated
condition = Đã sử dụng (chưa sửa chữa)
warranty_status = Expired
```

Parser CPU giả định tạo:

```text
cpu_brand = Intel
cpu_family_from_raw = Intel Core i5
cpu_model_code_group = Intel Gen 12 / U
```

Encoded:

```text
ram_gb = 16
storage_gb = 512
screen_size_inch = 15.6

brand_Dell = 1
brand_segment_business = 1
brand_is_rare = 0

model_Inspiron = 1
model_is_rare = 0

cpu_brand_Intel = 1
cpu_family_group_Intel Core i = 1
cpu_family_ord_filled = 2
cpu_intel_generation_ord = 12
cpu_amd_generation_ord = 0
cpu_apple_core_spec = 0
cpu_suffix_power_ord_filled = 1
cpu_tier_encoded = 6              # Mid-range

gpu_tier_ord_filled = 0
no_info_gpu = 0

condition_score = 2
warranty_expired = 1
warranty_active = 0
warranty_not_activated = 0

ram_storage_product_scaled = 1.0
ram_storage_balance = 1.0
is_mid_memory_storage = 1
```

---

## 15. Các feature không còn dùng trong final model

Các cột sau có thể xuất hiện ở dataset trung gian nhưng không thuộc final 86-feature schema:

```text
warranty_encoded
ram_level
storage_level
screen_size_level
brand_Gigabyte
brand_Other
brand_Sony
brand_Toshiba
model_Other
cpu_brand_Other
cpu_brand_Microsoft SQ
cpu_qualcomm_snapdragon_spec
gpu_type_Apple SoC
gpu_type_Dedicated
gpu_type_Integrated
gpu_type_Missing_Info
```

Khi build input production, nếu các cột này xuất hiện thì nên drop trước khi đưa vào model.

---

## 16. Checklist build input production

Trước khi predict, cần kiểm tra:

```text
[ ] Có đủ 86 feature theo final schema
[ ] Thứ tự cột đúng với final_laptop_price_feature_schema.json
[ ] Không còn raw text columns như brand/model/cpu/gpu
[ ] Không còn dropped columns như model_Other, brand_Other, gpu_type_*
[ ] Missing raw input được xử lý bằng NaN và missing/no-info flags
[ ] Brand/model không có trong final one-hot list không tạo cột mới
[ ] CPU tier map được audit lại nếu thay đổi rule tier
[ ] Extra columns được drop trước prediction
```

---

## 17. Kết luận

Final feature set không chỉ encode cấu hình phần cứng, mà còn encode thêm phân khúc thương hiệu, dòng máy, missing flags, CPU generation, CPU family, GPU tier, tình trạng máy, bảo hành và interaction giữa RAM/storage.

Điểm quan trọng nhất khi triển khai inference là không cố tạo lại tất cả cột từng có ở notebook 06. Chỉ tạo đúng các cột còn trong final schema. Các category bị notebook 11 loại bỏ thì không encode riêng nữa; nếu cần giữ tín hiệu, dùng các flag như `brand_is_rare`, `model_is_rare`, `brand_segment_rare` hoặc các missing/no-info flags.
