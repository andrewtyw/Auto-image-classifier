# Auto-image-classifier
### Prototype demo👇
[prototype deployment is here](http://120.24.230.237:81/)
### what is Auto-image-classifier?
Actually, the concept is quite simple. First you need to define the categories that you want to make a classification in `srape_train/fish.json`. Then the Web Scraper will get photos of these categories from Bing. Then train a model for classification.
### Setup
- Scrape & trian model
```shell
# first define the classification subjects in `/scrape_train/fish.json`
# then scrape web images of the subjects
cd scrape_train/
python fetch.py --limit 10  # the number of images for each class
# train model
python train_main.py 
```
- Deployment
```shell
cd flask_fish_reco/
python myapp.py
```

### File structure
`scrape_train/`: scraping photos and training.


`flask_fish_reco/`: is the web deployment with Flask.


`neural_vue/`: user interfaces using Vue.

### Prototype looks like 👉
<div align=center>
<img src="https://user-images.githubusercontent.com/78400045/203211338-5f531046-8d1d-4f3b-ba17-82e39379648c.png" width = "800" align=center />
</div>
<!-- ![image](https://user-images.githubusercontent.com/78400045/203211338-5f531046-8d1d-4f3b-ba17-82e39379648c.png)

![image](https://user-images.githubusercontent.com/78400045/151667687-79bbe512-6979-4fdf-94f4-a1ea2c635aa3.png) -->
