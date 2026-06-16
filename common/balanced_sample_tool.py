from typing import Union, Self

from pyspark.ml.feature import Bucketizer
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import numpy as np
import matplotlib.pyplot as plt
import math
import pyspark.sql.dataframe as ps

from common.utils import balanced_sample_theories, manual_sample_theories, balanced_sample_bins


class TheorySampler:
    """
    Class to sample theories evenly.
    """
    def __init__(self, filename: Union[str, None]=None):
        if filename is not None:
            self.filename = filename
            self.spark = SparkSession.builder.appName("CSVLoader").config("spark.driver.memory", "16g").config("spark.executor.memory", "16g").getOrCreate()
            self.df = self.spark.read.csv(
                filename,
                header=True,  # use first row as column names
                inferSchema=True,  # automatically infer column types
                sep=","  # delimiter (default is comma)
            )
        else:
            self.filename = None
            self.df = None

    def a_range(self) -> float:
        """
        Gets the range of a central charge
        :return: min_a, max_a
        """
        min_a = self.df.agg(F.min("CentralChargeA")).collect()[0][0]
        max_a = self.df.agg(F.max("CentralChargeA")).collect()[0][0]
        return min_a, max_a

    def c_range(self) -> float:
        """
        Gets the range of a central charge
        :return: min_c, max_c
        """
        min_c = self.df.agg(F.min("CentralChargeC")).collect()[0][0]
        max_c = self.df.agg(F.max("CentralChargeC")).collect()[0][0]
        return min_c, max_c

    def draw_central_charge_histogram(self, charge_interval: float) -> None:
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (16, 12)
        plt.rcParams['font.size'] = 15

        min_a, max_a = self.a_range()
        min_c, max_c = self.c_range()
        min_a = math.floor(min_a / charge_interval) * charge_interval
        min_c = math.floor(min_c / charge_interval) * charge_interval

        a_bins = math.ceil((max_a - min_a) / charge_interval)
        c_bins = math.ceil((max_c - min_c) / charge_interval)

        fig, ax = plt.subplots(1, 2, squeeze=True)
        fig.suptitle("Central Charge Distribution")

        ax[0].set_title("A charge")
        ax[0].hist(x=self.df.select("CentralChargeA").rdd.flatMap(lambda x: x).collect(), bins=a_bins, range=(min_a, min_a + charge_interval * a_bins))
        ax[0].set_xlabel("a")
        ax[0].set_ylabel("count")

        ax[1].set_title("C charge")
        ax[1].hist(x=self.df.select("CentralChargeA").rdd.flatMap(lambda x: x).collect(), bins=c_bins, range=(min_c, min_c + charge_interval * c_bins))
        ax[1].set_xlabel("c")
        ax[1].set_ylabel("count")

        plt.show()

    def get_theory_stats(self) -> ps.DataFrame:
        return self.df.groupBy("Name").count().orderBy(F.col("count").desc())

    def get_bins_stats(self, charge_col: str, min_charge: float, max_charge: float, n_bins: int) -> ps.DataFrame:
        filtered = self.df.filter((F.col(charge_col) >= min_charge) & (F.col(charge_col) < max_charge))

        bin_step = (max_charge - min_charge) / n_bins
        splits = [min_charge + i * bin_step for i in range(n_bins)] + [float("inf")]

        discretizer = Bucketizer(
            splits=splits,
            inputCol=charge_col,
            outputCol="bucket"
        )
        bucketed = discretizer.transform(filtered)

        bucket_counts = bucketed.groupBy("bucket").count()

        ranges = [(i, f"[{splits[i]}, {splits[i + 1]})") for i in range(len(splits) - 1)]
        ranges_df = self.spark.createDataFrame(ranges, ["bucket", "range"])

        bucket_summary = bucket_counts.join(ranges_df, on="bucket", how="inner")
        return bucket_summary

    def get_balanced_sample(self, a_range: tuple[float, float], c_range: tuple[float, float],
                            n_per_theory: int) -> Self:
        sample = TheorySampler()
        sample.filename = self.filename
        sample.spark = self.spark

        assert self.df is not None
        sample.df = balanced_sample_theories(self.df, "Name", "CentralChargeA", "CentralChargeC",
                                             a_range[0], a_range[1], c_range[0], c_range[1], n_per_theory)

        return sample

    def get_manual_sample(self, theories: list[str], n_per_theory: int) -> Self:
        sample = TheorySampler()
        sample.filename = self.filename
        sample.spark = self.spark

        assert self.df is not None
        sample.df = manual_sample_theories(self.df, "Name", "CentralChargeA", "CentralChargeC", theories, n_per_theory)

        return sample

    def get_theories_above_count(self, min_count: int) -> Self:
        sample = TheorySampler()
        sample.filename = self.filename
        sample.spark = self.spark

        assert self.df is not None
        stats = self.get_theory_stats()

        filtered = stats.filter(F.col("count") > min_count)
        names = filtered.select("Name").rdd.flatMap(lambda x: x).collect()

        sample.df = self.df.filter(F.col("Name").isin(names))

        return sample

    def get_selected_theories(self, theories: list[str]) -> Self:
        sample = TheorySampler()
        sample.filename = self.filename
        sample.spark = self.spark

        assert self.df is not None
        sample.df = self.df.filter(F.col("Name").isin(theories))

        return sample

    def get_balanced_bins_sample(self, charge_col: str, min_charge: float, max_charge: float, n_bins: int, n_per_bins: int) -> Self:
        sample = TheorySampler()
        sample.filename = self.filename
        sample.spark = self.spark

        assert self.df is not None
        sample.df = balanced_sample_bins(self.df, charge_col, min_charge, max_charge, n_bins, n_per_bins)

        return sample

    def get_theory_num(self) -> int:
        return self.df.select("Name").distinct().count()

    def __str__(self) -> str:
        return str(self.df)
