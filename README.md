Real estate in Kazan.

Start with downloading the data from kaggle[https://www.kaggle.com/datasets/dionicegenes/kazan-avito-2024-2026] and putting the table into data/raw folder. Images should be put inside data/images directory. Though you can customize structure in src/config.py
Then run scripts in src directory. In the following order: feature_engineering.py, preprocessing.py, model_training.py, text_extraction.py, image_embeddings.py, quantile_regression.py, xgboost_experiments.py. The last two scripts will save models needed to predict price, they are necessary for service to run. Other mentioned scipts process data.
Run service with docker:
- docker-compose up

Run service without docker:
- uvicorn backend.main:app --reload (for backend)
- npm run dev (in frontend directory)
