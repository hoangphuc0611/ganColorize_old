# Introduction

Trong thời đại 4.0 này việc chụp ảnh của mọi người dần trở nên thiết yếu và việc xảy ra lỗi trong khi chụp ảnh cũng không hề ít. Và khi mà người ta muốn phục chế lại những tấm ảnh đã cũ, cụ thể là những tấm ảnh trắng đen thì đã cho ra đời nhiều mô hình Deep learning nhằm giúp tô màu cho những tấm ảnh trắng đen đó. 

Đầu vào ban đầu: 1 tấm ảnh trắng đen

Đầu ra sau mô hình: 1 tấm ảnh đã được tô màu


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.
```bash
$ pip install -r requirements.txt
```
## Dataset
+ Trích 100.000 ảnh từ bộ dữ liệu CelebFace Attributes trên Kaggle, bao gồm những tấm ảnh của những người trưởng thành.
+ Crawl 11.937 tấm ảnh trên các trang mạng( pixabay.com, Yandex.com, istockphoto.com) và từ google gồm: 5940 ảnh em bé và trẻ con, 5997 ảnh người già.

## Model used
1.	Mô hình cGAN
a.	Kiến trúc model cGAN
Kiến trúc của cGAN bao gồm 2 mạng generator vaf discriminator
 
b.	Generator
•	Mục tiêu của Geneator là tạo ra được nội dung không thể phân biệt được với tập huấn luyện đến nỗi Discriminator không thể phân biệt được.
•	Kiến trúc của mô hình:

	 
c.	Discriminator
•	Mục tiêu của Discriminator là trở thành chuyên gia về hình ảnh để có thể phân biệt được đâu là ảnh thật đâu là ảnh giả. Nếu nó bị đánh lừa bởi Generator quá sớm thì nó sẽ không làm tốt công việc của mình và kết quả sẽ không thể đào tạo Generator tốt được








•	Kiến trúc mô hình
 


d.	Hàm loss

Loss_G  =  || y – G(x) ||^2 + L1 loss*lamda
Loss_D =  (real_loss + fake_loss)*0.5

## Train model

Select the location to save the model in file translate 
Save file model to ``checkpoint-trans/...``

## RUN test
```bash
$ flask run 
```

