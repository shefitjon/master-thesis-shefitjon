<?php
declare(strict_types=1);

namespace Bregu\CartRecoveryAb\Model;

class Config
{
    public const SOURCE_LIVE = 'live';
    public const SOURCE_PREGENERATED = 'pregenerated';

    private const XML_PATH_ENABLED = 'cart_recovery_ab/general/enabled';
    private const XML_PATH_SPLIT = 'cart_recovery_ab/general/ab_split';
    private const XML_PATH_THRESHOLD = 'cart_recovery_ab/general/risk_threshold';
    private const XML_PATH_SOURCE = 'cart_recovery_ab/general/intervention_source';
    private const XML_PATH_TRIGGER = 'cart_recovery_ab/general/trigger';
    private const XML_PATH_API_KEY = 'cart_recovery_ab/general/gemini_api_key';

    public function __construct(
        private readonly \Magento\Framework\App\Config\ScopeConfigInterface $scopeConfig,
        private readonly \Magento\Framework\Encryption\EncryptorInterface $encryptor
    ) {
    }

    public function isEnabled(): bool
    {
        return $this->scopeConfig->isSetFlag(
            self::XML_PATH_ENABLED,
            \Magento\Store\Model\ScopeInterface::SCOPE_STORE
        );
    }

    public function getTreatmentShare(): int
    {
        $value = (int) $this->scopeConfig->getValue(
            self::XML_PATH_SPLIT,
            \Magento\Store\Model\ScopeInterface::SCOPE_STORE
        );

        return max(0, min(100, $value));
    }

    public function getRiskThreshold(): float
    {
        return (float) $this->scopeConfig->getValue(
            self::XML_PATH_THRESHOLD,
            \Magento\Store\Model\ScopeInterface::SCOPE_STORE
        );
    }

    public function getInterventionSource(): string
    {
        return (string) $this->scopeConfig->getValue(
            self::XML_PATH_SOURCE,
            \Magento\Store\Model\ScopeInterface::SCOPE_STORE
        );
    }

    public function getTrigger(): string
    {
        return (string) $this->scopeConfig->getValue(
            self::XML_PATH_TRIGGER,
            \Magento\Store\Model\ScopeInterface::SCOPE_STORE
        );
    }

    public function getGeminiApiKey(): string
    {
        $stored = (string) $this->scopeConfig->getValue(
            self::XML_PATH_API_KEY,
            \Magento\Store\Model\ScopeInterface::SCOPE_STORE
        );

        if ($stored === '') {
            return '';
        }

        return (string) $this->encryptor->decrypt($stored);
    }
}
