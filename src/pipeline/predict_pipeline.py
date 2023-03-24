import sys
from src.exception import CustomException
from src.utils import load_object, main_pic_linker


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, anime, data, data_transformed, similarity_matrix):
        try:
            index = data_transformed[data_transformed['title'] == anime].index[0]
            distances = sorted(list(enumerate(similarity_matrix[index])),
                               reverse=True,
                               key=lambda x: x[1])

            result_dict = {'title': [],
                           'pic_url': [],
                           'synopsis': [],
                           'studio': [],
                           'type': [],
                           'num_episodes': [],
                           'status': [],
                           'score': []}

            for dist in distances:
                anime_id = data.iloc[dist[0]].anime_id
                result_dict['title'].append(data.loc[data['anime_id'] == anime_id, ['title']])
                result_dict['pic_url'].append(main_pic_linker(data, anime_id))
                result_dict['synopsis'].append(data.loc[data['anime_id'] == anime_id, ['synopsis']])
                result_dict['studio'].append(data.loc[data['anime_id'] == anime_id, ['studio']])
                result_dict['type'].append(data.loc[data['anime_id'] == anime_id, ['type']])
                result_dict['num_episodes'].append(data.loc[data['anime_id'] == anime_id, ['num_episodes']])
                result_dict['status'].append(data.loc[data['anime_id'] == anime_id, ['status']])
                result_dict['score'].append(data.loc[data['anime_id'] == anime_id, ['score']])

            return result_dict
        except Exception as e:
            raise CustomException(e, sys)
