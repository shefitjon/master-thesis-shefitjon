<?php
declare(strict_types=1);

namespace Bregu\CartRecoveryAb\Model\Config\Source;

class Trigger implements \Magento\Framework\Data\OptionSourceInterface
{
    public function toOptionArray(): array
    {
        return [
            ['value' => 'on_load', 'label' => __('On cart load')],
            ['value' => 'exit_intent', 'label' => __('On exit intent')],
            ['value' => 'idle', 'label' => __('After idle')],
        ];
    }
}
