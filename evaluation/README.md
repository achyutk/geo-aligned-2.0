
Install the following packages if you want execute only these code, If the environment is already set up using the requirements file from the mail directory, then this step can be avoided:

> pip install -U torch torchvision <br>
> pip install -U git+https://github.com/openai/CLIP.git <br>
> pip install pycountry

# download_wiki_images_eval

This code present in **/evaluation/data** used for extracting images from wikipedia. These images are later used for evaluating the framework. In the code , we iterate over the list of country names and download relevant images from the country's wikipedia page 

The code will create folders fro respective country in the **/evaluation/data/image/** folder and store images in it.


### country_text_features.py

This file will generate two files in **/evaluation/model** named  eval_country_text_features.pt and eval_country_text_country_list.txt 

> eval_country_text_features.pt stores features for text from clip model which is later used for evaluation <br>
> eval_country_text_country_list.txt stores the names of country used to generate the feature set


### country_img_features.py

This code is generating feature set for images sourced from Wikipedia which is stored in **/evaluation/data/image/** , which is later used for evaluation. It will generate two files in **/evaluation/model** named  eval_img_features.pt and eval_img_features_country_list.txt 

> eval_img_features.pt stores features for image dataset from clip model which is later used for evaluation <br>
> eval_img_features_country_list.txt stores the names of country used to generate the feature set

### eval.ipynb

This code will reach out to **/output/framework_name** folder for images to evaluate the framework. It will generate four files in the same location. Each representing the results on the four metrics whose functions are defined in the utils.py file. 
