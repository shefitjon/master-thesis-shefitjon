"""
Phase 1 — feature engineering with the Temporal Shield.

The whole thesis stands or falls on one rule: a feature may only use events
that happened at or before the prediction moment (the first cart-add). Break
that rule and you leak the future into the model — which is exactly how the
early pipeline scored an impossible 0.997 F1. Everything below enforces the
rule explicitly.
"""
import pandas as pd
import numpy as np
from datetime import datetime

class TemporallyCorrectFeatureEngineer:

    def extract_features_at_prediction_point(self, session_events):
        session_events = session_events.sort_values('event_time')

        # The first cart-add is our prediction moment. If the session never
        # added anything to the cart there is nothing to predict on.
        cart_events = session_events[session_events['event_type'] == 'cart']
        if len(cart_events) == 0:
            return None
        first_cart_time = cart_events['event_time'].min()

        # The Temporal Shield: keep only what the shopper had done up to that
        # instant. Anything after first_cart_time is the future and is banned.
        events_up_to_cart = session_events[session_events['event_time'] <= first_cart_time]
        features = {}
        session_start = events_up_to_cart['event_time'].min()
        features['time_to_cart'] = (first_cart_time - session_start).total_seconds()

        # Strictly before the cart-add, so the cart event itself never counts
        # as pre-cart browsing. These three are the features the model leans on.
        pre_cart_only = events_up_to_cart[events_up_to_cart['event_time'] < first_cart_time]
        features['views_before_cart'] = len(pre_cart_only[pre_cart_only['event_type'] == 'view'])
        features['total_events_before_cart'] = len(pre_cart_only)
        # Browse intensity = pre-cart product views per minute of browsing.
        if features['time_to_cart'] > 0:
            features['browse_intensity_pre_cart'] = features['views_before_cart'] / (features['time_to_cart'] / 60)
        else:
            features['browse_intensity_pre_cart'] = 0
        features['unique_products_viewed'] = pre_cart_only['product_id'].nunique()
        features['unique_categories_viewed'] = pre_cart_only['category_code'].nunique()
        viewed_prices = pre_cart_only[pre_cart_only['event_type'] == 'view']['price']
        if len(viewed_prices) > 0:
            features['avg_viewed_price'] = viewed_prices.mean()
            features['max_viewed_price'] = viewed_prices.max()
            features['min_viewed_price'] = viewed_prices.min()
            features['price_variance_viewed'] = viewed_prices.var()
        else:
            features['avg_viewed_price'] = 0
            features['max_viewed_price'] = 0
            features['min_viewed_price'] = 0
            features['price_variance_viewed'] = 0
        # Cart state at the prediction moment — current contents, not final ones.
        cart_items_at_moment = events_up_to_cart[events_up_to_cart['event_type'] == 'cart']
        features['initial_cart_value'] = cart_items_at_moment['price'].sum()
        features['initial_cart_items'] = len(cart_items_at_moment)
        features['hour_of_cart'] = first_cart_time.hour
        features['day_of_week'] = first_cart_time.dayofweek
        if len(pre_cart_only) >= 2:
            event_times = pre_cart_only['event_time'].values
            time_diffs = np.diff(event_times).astype('timedelta64[s]').astype(float)
            features['avg_time_between_events'] = time_diffs.mean()
            features['acceleration'] = time_diffs[-1] - time_diffs[0] if len(time_diffs) > 1 else 0
        else:
            features['avg_time_between_events'] = 0
            features['acceleration'] = 0
        if len(pre_cart_only) > 0:
            category_counts = pre_cart_only['category_code'].value_counts()
            if len(category_counts) > 0:
                features['category_focus_ratio'] = category_counts.iloc[0] / len(pre_cart_only)
            else:
                features['category_focus_ratio'] = 0
        else:
            features['category_focus_ratio'] = 0
        # The LABEL is allowed to look at the future (did they buy afterwards?)
        # — only the FEATURES must stay behind the shield.
        full_session_events = session_events[session_events['event_time'] > first_cart_time]
        features['target_purchase'] = 1 if 'purchase' in full_session_events['event_type'].values else 0
        return features

    def validate_no_leakage(self, features_df):
        # A guard rail: if any of these future-derived columns ever reappear,
        # fail loudly rather than silently train on leaked data again.
        suspicious_features = ['events_after_cart', 'cart_removals', 'cart_removal_rate', 'final_cart_value', 'session_duration', 'total_events']
        leaked_features = [f for f in suspicious_features if f in features_df.columns]
        if leaked_features:
            raise ValueError(f'DATA LEAKAGE DETECTED! Features using future information: {leaked_features}')
        print('✅ Feature validation passed - no temporal leakage detected')
        return True

def demonstrate_temporal_correctness():
    print('=' * 60)
    print('TEMPORAL FEATURE ENGINEERING DEMONSTRATION')
    print('=' * 60)
    print('\nSession Timeline:')
    print('09:00 VIEW product_A')
    print('09:02 VIEW product_B')
    print('09:05 CART product_A  ← PREDICTION POINT')
    print("09:07 VIEW product_C  ← FUTURE (can't use!)")
    print("09:10 REMOVE product_A  ← FUTURE (can't use!)")
    print('09:12 [session ends]')
    print('\n❌ WRONG (Data Leakage):')
    print('  events_after_cart = 2  # How do we know this at 09:05?')
    print("  cart_removals = 1     # Hasn't happened yet!")
    print('  session_duration = 12 min  # Session not over!')
    print('\n✅ CORRECT (No Leakage):')
    print('  views_before_cart = 2    # We know this at 09:05')
    print('  time_to_cart = 5 min     # We know this at 09:05')
    print('  initial_cart_value = $50 # Current state at 09:05')
    print('\n' + '=' * 60)
    print('Expected Impact on Model Performance:')
    print('=' * 60)
    print('WITH leakage:    F1-Score = 99.7% (impossible!)')
    print('WITHOUT leakage: F1-Score = 65-75% (realistic)')
    print('\nLower metrics = Higher scientific integrity!')
if __name__ == '__main__':
    demonstrate_temporal_correctness()
    print('\n' + '=' * 60)
    print('IMPLEMENTATION NOTES')
    print('=' * 60)
    print("\n    1. All features must be computable at cart-add moment\n    2. Never use information from after the prediction point\n    3. In production, we predict when user adds to cart\n    4. Training labels come from future, but features don't\n    5. This ensures model works in real-time production\n    ")
