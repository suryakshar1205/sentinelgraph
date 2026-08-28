"""
Leakage-safe unified feature extraction pipeline for SentinelGraph.
"""

import pandas as pd
from typing import Tuple, List
from features.behavioral import BehavioralFeatureExtractor
from features.entity_features import EntityFeatureExtractor
from features.temporal import TemporalFeatureExtractor


class FeaturePipeline:
    """Master pipeline producing isolated feature matrices for Baseline and SentinelGraph."""

    def __init__(self):
        self.behavioral_extractor = BehavioralFeatureExtractor()
        self.entity_extractor = EntityFeatureExtractor()
        self.temporal_extractor = TemporalFeatureExtractor()

    def extract_baseline_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract only transaction & customer behavioral features for the baseline model.
        Explicitly excludes graph relationships and cross-entity sharing.
        """
        return self.behavioral_extractor.extract_features(df)

    def extract_full_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract behavioral, entity-graph, and temporal feature sets.
        Returns (behavioral_df, entity_df, temporal_df).
        """
        df_sorted = df.sort_values(by="timestamp_unix").reset_index(drop=True)
        
        beh_df = self.behavioral_extractor.extract_features(df_sorted)
        ent_df = self.entity_extractor.extract_features(df_sorted)
        tem_df = self.temporal_extractor.extract_features(df_sorted)
        
        return beh_df, ent_df, tem_df


if __name__ == "__main__":
    df = pd.read_parquet("data/generated/train.parquet").head(1000)
    pipeline = FeaturePipeline()
    b, e, t = pipeline.extract_full_features(df)
    print(f"Behavioral features: {b.shape}")
    print(f"Entity features:     {e.shape}")
    print(f"Temporal features:   {t.shape}")
