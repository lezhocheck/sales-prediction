from typing import Optional
import numpy as np
import polars as pl
import os
from holidays import country_holidays
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class DatasetGenerator:
    NUM_LAGS: int = 3
    WINDOW_SIZE: int = 5
    NUM_CLUSTERS: int = 5

    _scaler: Optional[StandardScaler] = None
    _kmeans: Optional[KMeans] = None

    def __init__(
        self, 
        src_data_path: str, 
        generate_lags: bool = True,
        generate_sma: bool = True,
        generate_clusters: bool = True,
        remove_nulls: bool = True,
        apply_norm: bool = False
    ) -> None:
        
        path = os.path.abspath(src_data_path)
        assert os.path.isfile(path), f'Specified file {src_data_path} does not exist'
        holidays = country_holidays('UA')
        self._data = pl.read_csv(
            src_data_path, 
            columns=['date', 'category_id', 'sku_id', 'sales_price', 'sales_quantity']
        ).with_columns(pl.col('date').str.to_date()).with_columns(
            pl.col('date').dt.year().alias('year'),
            pl.col('date').dt.month().alias('month'),
            pl.col('date').dt.day().alias('day'),
            pl.col('date').dt.weekday().alias('weekday'),
            pl.col('date').is_in(set(holidays)).alias('is_ukrainian_holiday')
        )
        if generate_lags:
            self._data = self._generate_lags(self._data)
        if generate_sma:
            self._data = self._generate_sma(self._data)
        if generate_clusters:
            self._data = self._generate_cluster_id_feature(self._data)
        if remove_nulls:
            self._data = self._data.drop_nulls()
        if apply_norm:
            schema = self._data.schema
            DatasetGenerator._scaler = StandardScaler()
            self._data = DatasetGenerator._scaler.fit_transform(self._data)
            self._data = pl.DataFrame(self._data, schema={i: pl.Float64 for i in schema.keys()})
        self._data = self._data.drop('date').sample(fraction=1, shuffle=True)
    
    @property
    def normalizer(self) -> StandardScaler:
        if self._scaler is None:
            raise ValueError('Normalization was not applied to this data')
        return self._scaler

    @property
    def cluster_generator(self) -> KMeans:
        if self._kmeans is None:
            raise ValueError('Clustering was not applied to this data')
        return self._kmeans

    @classmethod
    def _generate_lags(cls, data: pl.DataFrame) -> pl.DataFrame:
        lag_agg_sku_id = data.group_by(['sku_id', 'date']).agg(
            pl.col('sales_price').mean(),
            pl.col('sales_quantity').sum()
        ).sort(by=['sku_id', 'date'])
        cols_sales_price = [pl.col('sales_price').shift(lag).alias(f'lag_{lag}_sales_price') for lag in range(1, cls.NUM_LAGS+1)]
        cols_sales_quantity = [pl.col('sales_quantity').shift(lag).alias(f'lag_{lag}_sales_quantity') for lag in range(1, cls.NUM_LAGS+1)]
        aggs = cols_sales_price + cols_sales_quantity
        aggs.insert(0, pl.col('date'))
        lag_agg_sku_id = lag_agg_sku_id.group_by('sku_id', maintain_order=True).agg(aggs)
        cols = [col for col in lag_agg_sku_id.columns if col != 'sku_id']
        merged_skus = []
        for sku_id, date, *aggs in lag_agg_sku_id.iter_rows():
            merged_skus.append(
                pl.DataFrame(
                    schema=cols,
                    data=[date, *aggs], orient='col'
                ).with_columns(pl.lit(sku_id).cast(pl.Int64).alias('sku_id'))
            )
        return data.join(pl.concat(merged_skus), on=['sku_id', 'date'])

    @classmethod
    def _generate_sma(cls, data: pl.DataFrame) -> pl.DataFrame:
        categories_agg = data.group_by(['category_id', 'date']).agg(
            pl.col('sales_price').mean(),
            pl.col('sales_quantity').sum()
        ).sort(by=['category_id', 'date'])
        sma_aggregation = categories_agg.group_by(['category_id'], maintain_order=True).agg(
            pl.col('date'),
            pl.col('sales_price').rolling_mean(window_size=cls.WINDOW_SIZE),
            pl.col('sales_quantity').rolling_mean(window_size=cls.WINDOW_SIZE)
        )
        sma_merged = []
        for category, date, sale_price, sales_quantity in sma_aggregation.iter_rows():
            sma_merged.append(
                pl.DataFrame(
                    schema=['date', f'sma_{cls.WINDOW_SIZE}_sales_price', f'sma_{cls.WINDOW_SIZE}_sales_quantity'],
                    data=[date, sale_price, sales_quantity], orient='col'
                ).with_columns(pl.lit(category).cast(pl.Int64).alias('category_id'))
            )
        return data.join(pl.concat(sma_merged, how='vertical'), on=['category_id', 'date'])

    @classmethod
    def _generate_cluster_id_feature(cls, data: pl.DataFrame) -> pl.DataFrame:
        mask = np.any(data.select(pl.all().is_null()).to_numpy(), axis=1)
        nulls = data.filter(pl.Series(mask))
        features = data.drop_nulls()
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
        cls._kmeans = KMeans(n_clusters=cls.NUM_CLUSTERS)
        clusters = cls._kmeans.fit_predict(scaled)
        processed_dataset = features.with_columns(
            pl.Series(values=clusters).alias('cluster_id')
        )
        return pl.concat([processed_dataset, nulls.with_columns(pl.lit(None).alias('cluster_id'))])