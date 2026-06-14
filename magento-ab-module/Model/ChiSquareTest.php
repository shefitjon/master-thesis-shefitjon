<?php
declare(strict_types=1);

namespace Bregu\CartRecoveryAb\Model;

class ChiSquareTest
{
    /**
     * 2x2 chi-square test of conversion with Yates' continuity correction.
     *
     * @return array{chi2: float, p: float}
     */
    public function test(
        int $controlConversions,
        int $controlTotal,
        int $treatmentConversions,
        int $treatmentTotal
    ): array {
        $a = $treatmentConversions;
        $b = $treatmentTotal - $treatmentConversions;
        $c = $controlConversions;
        $d = $controlTotal - $controlConversions;
        $n = $a + $b + $c + $d;

        $rowTreatment = $a + $b;
        $rowControl = $c + $d;
        $colConverted = $a + $c;
        $colNot = $b + $d;

        if ($n === 0 || $rowTreatment === 0 || $rowControl === 0 || $colConverted === 0 || $colNot === 0) {
            return ['chi2' => 0.0, 'p' => 1.0];
        }

        $corrected = max(0.0, abs(($a * $d) - ($b * $c)) - ($n / 2.0));
        $chi2 = ($n * $corrected * $corrected) / ($rowTreatment * $rowControl * $colConverted * $colNot);
        $p = $this->erfc(sqrt($chi2 / 2.0));

        return ['chi2' => $chi2, 'p' => $p];
    }

    /**
     * Complementary error function (Abramowitz & Stegun 7.1.26, |error| < 1.5e-7).
     * For 1 degree of freedom, P(chi2 > x) = erfc(sqrt(x / 2)).
     */
    private function erfc(float $x): float
    {
        $z = abs($x);
        $t = 1.0 / (1.0 + 0.5 * $z);
        $ans = $t * exp(
            -$z * $z - 1.26551223
            + $t * (1.00002368
            + $t * (0.37409196
            + $t * (0.09678418
            + $t * (-0.18628806
            + $t * (0.27886807
            + $t * (-1.13520398
            + $t * (1.48851587
            + $t * (-0.82215223
            + $t * 0.17087277))))))))
        );

        return $x >= 0.0 ? $ans : 2.0 - $ans;
    }
}
