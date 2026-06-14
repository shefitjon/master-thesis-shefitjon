<?php
declare(strict_types=1);

namespace Bregu\CartRecoveryAb\Model;

class RiskScorer
{
    private const INTERCEPT = -0.023951112675070718;

    private const FEATURES = [
        'views_before_cart' => ['coef' => -5.227051346293989, 'mean' => 3.374099, 'scale' => 5.580918],
        'total_events_before_cart' => ['coef' => 5.137406232407244, 'mean' => 3.389909, 'scale' => 5.594586],
        'browse_intensity_pre_cart' => ['coef' => 0.20650356322302052, 'mean' => 3.055820, 'scale' => 5.735370],
    ];

    /**
     * @return float P(abandon) in [0, 1]
     */
    public function abandonmentProbability(
        int $viewsBeforeCart,
        int $totalEventsBeforeCart,
        float $browseIntensity
    ): float {
        $logit = self::INTERCEPT
            + $this->contribution('views_before_cart', $viewsBeforeCart)
            + $this->contribution('total_events_before_cart', $totalEventsBeforeCart)
            + $this->contribution('browse_intensity_pre_cart', $browseIntensity);

        $probabilityPurchase = 1.0 / (1.0 + exp(-$logit));

        return 1.0 - $probabilityPurchase;
    }

    private function contribution(string $feature, float $value): float
    {
        $parameters = self::FEATURES[$feature];

        return $parameters['coef'] * (($value - $parameters['mean']) / $parameters['scale']);
    }
}
