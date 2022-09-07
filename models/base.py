from sdgym.synthesizers.base import *


# class BaseSynthesizer(LegacySingleTableBaseline):
#     """Single table baseline which passes ordinals and categoricals down.
#
#     This class exists here to support the legacy baselines which do not operate
#     on metadata and instead expect lists of categorical and ordinal columns.
#     """
#
#     MODALITIES = ('single-table', )
#
#     def _get_columns(self, real_data, table_metadata):
#         model_columns = []
#         categorical_columns = []
#
#         fields_meta = table_metadata['fields']
#
#         for column in real_data.columns:
#             field_meta = fields_meta[column]
#             field_type = field_meta['type']
#             if field_type == 'id':
#                 continue
#
#             index = len(model_columns)
#             if field_type == 'categorical':
#                 categorical_columns.append(index)
#
#             model_columns.append(column)
#
#         return model_columns, categorical_columns
#
#     def _fit_sample(self, real_data, table_metadata):
#         columns, categoricals = self._get_columns(real_data, table_metadata)
#         real_data = real_data[columns]
#
#         ht = rdt.HyperTransformer(dtype_transformers={
#             'O': 'label_encoding',
#         })
#         ht.fit(real_data.iloc[:, categoricals])
#         model_data = ht.transform(real_data)[columns]
#
#         supported = set(model_data.select_dtypes(('number', 'bool')).columns)
#         unsupported = set(model_data.columns) - supported
#         if unsupported:
#             unsupported_dtypes = model_data[unsupported].dtypes.unique().tolist()
#             raise UnsupportedDataset(f'Unsupported dtypes {unsupported_dtypes}')
#
#         nulls = model_data.isnull().any()
#         if nulls.any():
#             unsupported_columns = nulls[nulls].index.tolist()
#             raise UnsupportedDataset(f'Null values found in columns {unsupported_columns}')
#
#         LOGGER.info("Fitting %s", self.__class__.__name__)
#         self.fit(model_data.to_numpy(), categoricals, ())
#
#         LOGGER.info("Sampling %s", self.__class__.__name__)
#         sampled_data = self.sample(len(model_data))
#         sampled_data = pd.DataFrame(sampled_data, columns=columns)
#
#         # synthetic_data = real_data.copy()
#         synthetic_data = pd.DataFrame(ht.reverse_transform(sampled_data))[columns]
#         synthetic_data.to_csv("temp.csv")
#         # synthetic_data.update(ht.reverse_transform(sampled_data))
#         return synthetic_data

# def write_epoch_data(data, columns, meta, output_filename):
#     # # print(data.shape)
#     # print(meta['continuous_columns'])
#     # print(meta['discrete_columns'])
#     # print(meta['column_info'])
#     # exit()
#     dataframe = pd.DataFrame(data, columns=columns)
#     dataframe.to_csv(output_filename, index=False)

import logging

LOGGER = logging.getLogger(__name__)


class BaseSynthesizer:
    """Base class for all default synthesizers of ``SDGym``."""

    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        pass

    def sample(self, samples):
        pass

    def fit_sample(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        LOGGER.info("Fitting %s", self.__class__.__name__)
        self.fit(data, categorical_columns, ordinal_columns)

        LOGGER.info("Sampling %s", self.__class__.__name__)
        data = self.sample(data.shape[0])

        return data
        # return self.sample(data.shape[0])