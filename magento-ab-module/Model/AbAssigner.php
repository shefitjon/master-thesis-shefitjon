<?php
declare(strict_types=1);

namespace Bregu\CartRecoveryAb\Model;

class AbAssigner
{
    public const BUCKET_CONTROL = 'control';
    public const BUCKET_TREATMENT = 'treatment';

    public function __construct(
        private readonly \Bregu\CartRecoveryAb\Model\Config $config
    ) {
    }

    public function bucketFor(string $sessionKey): string
    {
        $position = abs(crc32($sessionKey)) % 100;

        return $position < $this->config->getTreatmentShare()
            ? self::BUCKET_TREATMENT
            : self::BUCKET_CONTROL;
    }
}
