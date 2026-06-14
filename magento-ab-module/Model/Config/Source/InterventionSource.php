<?php
declare(strict_types=1);

namespace Bregu\CartRecoveryAb\Model\Config\Source;

class InterventionSource implements \Magento\Framework\Data\OptionSourceInterface
{
    public function toOptionArray(): array
    {
        return [
            ['value' => 'live', 'label' => __('Live Gemini')],
            ['value' => 'pregenerated', 'label' => __('Pre-generated bank')],
        ];
    }
}
