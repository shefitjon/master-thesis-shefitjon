"""
Phase 1 — data engineering: raw REES46 events -> leakage-free session features.

This is the full pipeline that builds data/engineered_sessions_no_leakage.csv:
  1. Stream the 42.4M raw October-2019 events in 500k-row chunks (they do not fit
     in memory) and classify every cart session as abandon or purchase.
  2. Sample a fixed 100k sessions at a 70/30 abandon/purchase split, to match the
     ~70% Baymard industry cart-abandonment rate.
  3. Engineer 20 features per session under the Temporal Shield — every feature is
     computed only from events strictly before the first cart-add (the prediction
     moment); only the label is allowed to look at the future.

The raw 2019-Oct.csv is gitignored (download separately — see README). The shipped
data/engineered_sessions_no_leakage.csv is the canonical dataset every later phase
trains on. (src/phase1_feature_engineering.py is a focused, runnable demo of just
the shield logic; this file is the end-to-end pipeline.)
"""
import warnings
from pathlib import Path
from typing import Tuple
import random
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

SEED = 42
# Deterministic sampling. Note: the shipped CSV was produced before this seed was
# pinned, so a fresh run reproduces the same 70/30 distribution, not identical rows.
random.seed(SEED)

HERE = Path(__file__).resolve().parents[1]
RAW_CSV = HERE / 'data' / '2019-Oct.csv'
OUT_CSV = HERE / 'data' / 'engineered_sessions_no_leakage.csv'


class TemporallyCorrectDataEngineer:

    def __init__(self, data_path: Path, target_sessions: int = 100000):
        self.data_path = data_path
        self.target_sessions = target_sessions
        # 42.4M events do not fit in memory — read in chunks.
        self.chunk_size = 500000
        self.cart_abandon_sessions = set()
        self.cart_purchase_sessions = set()

    def find_cart_sessions_with_outcomes(self) -> Tuple[set, set]:
        """Pass 1: label every cart session by its eventual outcome."""
        print('PHASE 1a — identifying cart sessions')
        chunks_processed = 0
        for chunk in pd.read_csv(self.data_path, chunksize=self.chunk_size,
                                 usecols=['event_type', 'user_session']):
            for session_id, group in chunk.groupby('user_session'):
                events = set(group['event_type'].values)
                # A cart session is an abandoner unless a purchase also appears.
                if 'cart' in events:
                    if 'purchase' in events:
                        self.cart_purchase_sessions.add(session_id)
                    else:
                        self.cart_abandon_sessions.add(session_id)
            chunks_processed += 1
            if chunks_processed % 10 == 0:
                total_cart = len(self.cart_abandon_sessions) + len(self.cart_purchase_sessions)
                if total_cart > 0:
                    abandon_rate = len(self.cart_abandon_sessions) / total_cart * 100
                    print(f'  {chunks_processed * self.chunk_size:,} rows — {total_cart:,} cart sessions ({abandon_rate:.1f}% abandon)')
            # Stop once we have comfortably more than we need to sample from.
            if len(self.cart_abandon_sessions) + len(self.cart_purchase_sessions) > self.target_sessions * 2:
                break
        print(f'  found {len(self.cart_abandon_sessions):,} abandon / {len(self.cart_purchase_sessions):,} purchase sessions')
        return self.cart_abandon_sessions, self.cart_purchase_sessions

    def sample_balanced_sessions(self, abandon_sessions: set, purchase_sessions: set) -> set:
        """Force a 70/30 abandon/purchase split to match the ~70% Baymard rate."""
        print(f'PHASE 1b — sampling {self.target_sessions:,} sessions at 70/30')
        target_abandon = int(self.target_sessions * 0.70)
        target_purchase = int(self.target_sessions * 0.30)
        abandon_sample = min(target_abandon, len(abandon_sessions))
        purchase_sample = min(target_purchase, len(purchase_sessions))
        sampled_abandon = set(random.sample(list(abandon_sessions), abandon_sample))
        sampled_purchase = set(random.sample(list(purchase_sessions), purchase_sample))
        all_sampled = sampled_abandon.union(sampled_purchase)
        rate = len(sampled_abandon) / len(all_sampled) * 100
        print(f'  {len(all_sampled):,} sessions — {rate:.1f}% abandon / {100 - rate:.1f}% purchase')
        return all_sampled

    def extract_temporally_correct_features(self, sampled_sessions: set) -> pd.DataFrame:
        """Pass 2: pull the sampled sessions' events and build the 20 shielded features."""
        print('PHASE 1c — extracting events for sampled sessions')
        filtered_chunks = []
        chunks_processed = 0
        total_events = 0
        for chunk in pd.read_csv(self.data_path, chunksize=self.chunk_size):
            filtered = chunk[chunk['user_session'].isin(sampled_sessions)]
            if len(filtered) > 0:
                filtered_chunks.append(filtered)
                total_events += len(filtered)
            chunks_processed += 1
            if chunks_processed % 10 == 0:
                print(f'  {chunks_processed * self.chunk_size:,} rows — {total_events:,} events extracted')
            if total_events > len(sampled_sessions) * 20:
                break

        raw_df = pd.concat(filtered_chunks, ignore_index=True)
        raw_df['event_time'] = pd.to_datetime(raw_df['event_time'])
        print(f'  {len(raw_df):,} events — engineering features under the Temporal Shield')

        engineered_data = []
        sessions_processed = 0
        for session_id, group in raw_df.groupby('user_session'):
            group = group.sort_values('event_time')

            # The first cart-add is the prediction moment.
            cart_events = group[group['event_type'] == 'cart']
            if len(cart_events) == 0:
                continue
            first_cart_time = cart_events['event_time'].min()

            # The Temporal Shield: features may only see events up to / before that moment.
            events_up_to_cart = group[group['event_time'] <= first_cart_time]
            pre_cart_only = group[group['event_time'] < first_cart_time]

            features = {'session_id': session_id}

            # Temporal
            session_start = events_up_to_cart['event_time'].min()
            features['time_to_cart'] = (first_cart_time - session_start).total_seconds()
            features['hour_of_cart'] = first_cart_time.hour
            features['day_of_week'] = first_cart_time.dayofweek

            # Pre-cart behaviour (the features the model leans on)
            features['views_before_cart'] = len(pre_cart_only[pre_cart_only['event_type'] == 'view'])
            features['total_events_before_cart'] = len(pre_cart_only)
            if features['time_to_cart'] > 0:
                features['browse_intensity_pre_cart'] = features['views_before_cart'] / (features['time_to_cart'] / 60)
            else:
                features['browse_intensity_pre_cart'] = 0

            # Diversity of what was browsed
            features['unique_products_viewed'] = pre_cart_only['product_id'].nunique()
            features['unique_categories_viewed'] = pre_cart_only['category_code'].nunique()
            features['unique_brands_viewed'] = pre_cart_only['brand'].nunique()

            # Price profile of viewed items
            viewed_prices = pre_cart_only[pre_cart_only['event_type'] == 'view']['price']
            if len(viewed_prices) > 0:
                features['avg_viewed_price'] = viewed_prices.mean()
                features['max_viewed_price'] = viewed_prices.max()
                features['min_viewed_price'] = viewed_prices.min()
                features['price_variance_viewed'] = viewed_prices.var() if len(viewed_prices) > 1 else 0
            else:
                features['avg_viewed_price'] = 0
                features['max_viewed_price'] = 0
                features['min_viewed_price'] = 0
                features['price_variance_viewed'] = 0

            # Cart state at the prediction moment (current contents, not final)
            cart_items_at_moment = events_up_to_cart[events_up_to_cart['event_type'] == 'cart']
            features['initial_cart_value'] = cart_items_at_moment['price'].sum()
            features['initial_cart_items'] = len(cart_items_at_moment)
            features['avg_cart_item_price'] = cart_items_at_moment['price'].mean() if len(cart_items_at_moment) > 0 else 0

            # Browsing momentum
            if len(pre_cart_only) >= 2:
                event_times = pd.to_datetime(pre_cart_only['event_time'].values)
                time_diffs = np.diff(event_times).astype('timedelta64[s]').astype(float)
                features['avg_time_between_events'] = time_diffs.mean() if len(time_diffs) > 0 else 0
                features['event_acceleration'] = time_diffs[-1] - time_diffs[0] if len(time_diffs) > 1 else 0
            else:
                features['avg_time_between_events'] = 0
                features['event_acceleration'] = 0

            # Category focus + dominant category
            if len(pre_cart_only) > 0:
                category_counts = pre_cart_only['category_code'].value_counts()
                if len(category_counts) > 0:
                    features['category_focus_ratio'] = category_counts.iloc[0] / len(pre_cart_only)
                    features['main_category'] = str(category_counts.index[0]).split('.')[0] if pd.notna(category_counts.index[0]) else 'unknown'
                else:
                    features['category_focus_ratio'] = 0
                    features['main_category'] = 'unknown'
            else:
                features['category_focus_ratio'] = 0
                features['main_category'] = 'unknown'

            # The label is the one place the future is allowed: did a purchase follow?
            future_events = group[group['event_time'] > first_cart_time]
            features['target_purchase'] = 1 if 'purchase' in future_events['event_type'].values else 0

            engineered_data.append(features)
            sessions_processed += 1
            if sessions_processed % 5000 == 0:
                print(f'  {sessions_processed:,} sessions engineered')

        df_final = pd.DataFrame(engineered_data)
        self.validate_no_leakage(df_final)
        abandon_rate = (1 - df_final['target_purchase'].mean()) * 100
        print(f'  done — {len(df_final):,} sessions, {len(df_final.columns) - 2} features, {abandon_rate:.1f}% abandon')
        return df_final

    def validate_no_leakage(self, df: pd.DataFrame) -> bool:
        """Backstop: fail loudly if any known future-derived column slips back in."""
        suspicious = ['events_after_cart', 'cart_removals', 'cart_removal_rate',
                      'final_cart_value', 'session_duration', 'total_events',
                      'did_purchase', 'checkout_time']
        leaked = [f for f in suspicious if f in df.columns]
        if leaked:
            raise ValueError(f'DATA LEAKAGE DETECTED — features using the future: {leaked}')
        print('  temporal validation passed — no leakage')
        return True

    def save_clean_data(self, df: pd.DataFrame, output_path: Path = OUT_CSV) -> Path:
        df.to_csv(output_path, index=False)
        print(f'  saved {output_path}')
        return output_path


def run_temporally_correct_pipeline():
    engineer = TemporallyCorrectDataEngineer(data_path=RAW_CSV, target_sessions=100000)
    abandon_sessions, purchase_sessions = engineer.find_cart_sessions_with_outcomes()
    sampled_sessions = engineer.sample_balanced_sessions(abandon_sessions, purchase_sessions)
    clean_df = engineer.extract_temporally_correct_features(sampled_sessions)
    output_file = engineer.save_clean_data(clean_df)
    return clean_df, output_file


if __name__ == '__main__':
    run_temporally_correct_pipeline()
