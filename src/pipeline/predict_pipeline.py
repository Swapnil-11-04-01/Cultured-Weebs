import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object, main_pic_linker


class PredictPipeline:
    def __init__(self):
        pass

    @staticmethod
    def predict(anime, data, data_transformed, similarity_matrix):
        try:
            index = data_transformed[data_transformed['title'] == anime].index[0]
            distances = sorted(list(enumerate(similarity_matrix[index])),
                               reverse=True,
                               key=lambda x: x[1])

            result_dict = {'title': [],
                           'pic_url': [],
                           'genres': [],
                           'synopsis': [],
                           'studio': [],
                           'type': [],
                           'num_episodes': [],
                           'status': [],
                           'score': [],
                           'start_date': [],
                           'end_date': []}
            print(data.shape, data_transformed.shape)
            for dist in distances[1:16]:
                anime_id = data_transformed.iloc[dist[0]].anime_id
                result_dict['title'].append(data.loc[data['anime_id'] == anime_id, ['title']].iloc[0, 0])
                result_dict['pic_url'].append(main_pic_linker(data, anime_id))
                result_dict['genres'].append(data.loc[data['anime_id'] == anime_id, ['genres']].iloc[0, 0])
                result_dict['synopsis'].append(data.loc[data['anime_id'] == anime_id, ['synopsis']].iloc[0, 0])
                result_dict['studio'].append(data.loc[data['anime_id'] == anime_id, ['studios']].iloc[0, 0])
                result_dict['type'].append(data.loc[data['anime_id'] == anime_id, ['type']].iloc[0, 0])
                result_dict['num_episodes'].append(data.loc[data['anime_id'] == anime_id, ['num_episodes']].iloc[0, 0])
                result_dict['status'].append(data.loc[data['anime_id'] == anime_id, ['status']].iloc[0, 0])
                result_dict['score'].append(data.loc[data['anime_id'] == anime_id, ['score']].iloc[0, 0])
                result_dict['start_date'].append(data.loc[data['anime_id'] == anime_id, ['start_date']].iloc[0, 0])
                result_dict['end_date'].append(data.loc[data['anime_id'] == anime_id, ['end_date']].iloc[0, 0])
            # print(result_dict)
            return result_dict
        except Exception as e:
            raise CustomException(e, sys)
